import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, world_size: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_partition = out_features // world_size
        
        self.weight = nn.Parameter(torch.Tensor(self.out_features_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, world_size: int, bias: bool = False):
        super().__init__()
        self.in_features_per_partition = in_features // world_size
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, self.in_features_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partial_output = F.linear(x, self.weight, self.bias)
        dist.all_reduce(partial_output)
        return partial_output