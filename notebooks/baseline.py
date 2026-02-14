import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    path = os.path.expanduser("./models/Qwen/Qwen3-0.6B/")
    
    # 1. Preparation
    # 1-1. Define Tokenizer
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(path)
    print(f"[Main] Tokenizer loaded")
    
    # 1-2. Define Model
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    print(f"[Main] Model loaded")
    
    # 1-3. Define input
    prompts: list[str] = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    
    # Transform prompts to model input format using the model's chat template.
    # With add_generation_prompt=True, the tokenizer will append the assistant
    # role prefix defined in the chat template, so that the prompt ends exactly
    # at the point where the assistant is expected to start generating.
    #
    # Concretely, the resulting string will consist of:
    #   - the serialized user message(s)
    #   - followed by the assistant role prefix (model-specific),
    # without any assistant content yet.
    # For example (template-dependent, illustrative only):
    # ['<|im_start|>user\nintroduce yourself<|im_end|>\n<|im_start|>assistant\n', '<|im_start|>user\nlist all prime numbers within 100<|im_end|>\n<|im_start|>assistant\n']
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    print(prompts)
    print(f"[Main] Prompts defined")
    
    # 2. Tokenize input
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    print(f"[Main] Prompts tokenized")
    
    # 3. Model generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs, # <- A dict {'input_ids':..., 'attention_mask':...} that is directly unpacked into the generate() function
            temperature=0.6,
            max_new_tokens=256,
            do_sample=True
        )
        # The output format: torch.Tensor([batch_size, input_length + new_token])
        
        
    # 4. Decode output
    for i, output_ids in enumerate(outputs):
        prompt_len = int(inputs["attention_mask"][i].sum().item())  # To prevent padding issues
        
        completion_ids = output_ids[prompt_len:] # <- only decode the newly generated tokens, excluding the prompt tokens
        
        completion = tokenizer.decode(
            completion_ids,
            skip_special_tokens=True, # <- skip special tokens like <|im_start|>, <|im_end|> in the decoded text
        )
        
        # Use !r to show the raw string with special characters
        print("\n")
        print(f"[Main] Prompt: {prompts[i]!r}")
        print(f"[Main] Completion: {completion!r}")
    
    
if __name__ == "__main__":
    main()