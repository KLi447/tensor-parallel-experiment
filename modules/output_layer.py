import torch
from torch import nn
from .linear import ColumnParallelLinear

class OutputLayer(nn.Module):
    def __init__(self, config, world_size: int):
        super().__init__()
        self.lm_head_ = ColumnParallelLinear(config.hidden_size, config.vocab_size, world_size, bias=False)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.lm_head_(data).float()