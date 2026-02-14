from lllm import LLM

from loguru import logger


def test_lllm():
    llm = LLM()
    logger.info("Finished.")

if __name__ == "__main__":
    test_lllm()