import torch.multiprocessing as mp
import torch


# Two worker types: producer and consumer
# Use mp.Queue to communicate between processes
def producer(queue: mp.Queue, ack: mp.Queue):
    queue.put(torch.tensor([1, 2, 3, 4]))
    print(f"[Producer] Data sent")
    response = ack.get()
    print(f"[Producer] Received response: {response}")
    
def consumer(queue: mp.Queue, ack: mp.Queue):
    data = queue.get()
    print(f"[Consumer] Received: {data}")
    ack.put(True)
    
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    queue = mp.Queue(); ack = mp.Queue()
    
    p = mp.Process(target=producer, args=(queue, ack, ))
    c = mp.Process(target=consumer, args=(queue, ack, ))
    
    p.start(); c.start();
    p.join(); c.join();