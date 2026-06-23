import torch, time
class GPUProfiler:
    def __init__(self): self.events = []
    def __enter__(self): torch.cuda.synchronize(); return self
    def __exit__(self, *a): torch.cuda.synchronize()
