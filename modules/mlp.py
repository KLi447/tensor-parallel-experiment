import torch
from torch import nn
import torch.nn.functional as F
from .linear import ColumnParallelLinear, RowParallelLinear

class MLP(nn.Module):
    def __init__(self, config, layer_id: int, rank: int, world_size: int):
        super().__init__()
        self.layer_id_ = layer_id
        self.gate_ = ColumnParallelLinear(config.hidden_size, config.intermediate_size, world_size, bias=False)
        self.down_ = RowParallelLinear(config.intermediate_size, config.hidden_size, world_size, bias=False)
        self.up_ = ColumnParallelLinear(config.hidden_size, config.intermediate_size, world_size, bias=False)

    def forward(self, data: torch.Tensor):
        w1 = self.gate_(data)
        w3 = self.up_(data)
        mlp_output = F.silu(w1) * w3
        return self.down_(mlp_output)