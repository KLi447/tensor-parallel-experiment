import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm_eps_ = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_dtype = data.dtype
        v = data.to(torch.float32).pow(2).mean(-1, keepdim=True)
        rv = data * torch.rsqrt(v + self.norm_eps_)
        return (self.weight * rv).to(input_dtype)