import torch
from typing import Optional

class MemoryBank:
    def __init__(self, max_size: int, dim: int, device: torch.device, dtype: torch.dtype):
        self.max_size = int(max_size)
        self.dim = int(dim)
        self.device = device
        self.dtype = dtype

        self.reps = torch.empty((0, dim), device=device, dtype=dtype)
        self.labels = torch.empty((0,), device=device, dtype=torch.long)
        self.forbidden = torch.empty((0,), device=device, dtype=torch.long)

    @torch.no_grad()
    def enqueue(self, reps: torch.Tensor, labels: torch.Tensor, forbidden: Optional[torch.Tensor] = None):
        if self.max_size <= 0:
            return
        reps = reps.detach()
        labels = labels.detach()

        if forbidden is None:
            forbidden = torch.zeros((labels.size(0),), device=labels.device, dtype=torch.long)
        else:
            forbidden = forbidden.detach()

        reps = reps.to(device=self.device, dtype=self.dtype)
        labels = labels.to(device=self.device, dtype=torch.long)
        forbidden = forbidden.to(device=self.device, dtype=torch.long)

        self.reps = torch.cat([self.reps, reps], dim=0)
        self.labels = torch.cat([self.labels, labels], dim=0)
        self.forbidden = torch.cat([self.forbidden, forbidden], dim=0)

        if self.reps.size(0) > self.max_size:
            overflow = self.reps.size(0) - self.max_size
            self.reps = self.reps[overflow:]
            self.labels = self.labels[overflow:]
            self.forbidden = self.forbidden[overflow:]

    def is_ready(self) -> bool:
        return self.reps.numel() > 0