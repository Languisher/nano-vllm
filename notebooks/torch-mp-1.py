# An example of torch.multiprocessing
import torch.multiprocessing as mp
import torch


# Define the worker function
def worker_function(rank: int, shared_tensor: torch.Tensor):
    print(f"[Worker] Process {rank} started.")
    
    # Use indices to avoid data RACE
    shared_tensor[rank] += 1 # Modify the tensor IN PLACE
    
# Only executed by the main process
if __name__ == "__main__":
    print(f"[Main] Main started")
    
    # In torch.multiprocessing, it is recommended to use SPAWN method as default
    # As it is safe when dealing with intermediate variables handled by CUDA
    mp.set_start_method("spawn") # Use spawn
    # mp.set_start_method("fork") # Not recommended
    # mp.set_start_method("forkserver") # Used in specfic condition
    
    tensor: torch.Tensor = torch.zeros(4) # Shape = [4]
    tensor.share_memory_() # Enables all processes to see the same data in memory
    
    num_proc: int = 4
    processes: list[mp.Process] = []
    
    # Spawn {num_proc} processes and make them run
    for rank in range(num_proc):
        p = mp.Process(target=worker_function,
                       args=(rank, tensor))
        p.start()
        processes.append(p)
        
    # Main processes wait until all sub-processes finish
    for p in processes:
        p.join()
        
    print(f"[Main] Updated tensor: {tensor}.")