import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import MutableMapping

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, world_size: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_partition = out_features // world_size
        
        self.weight = nn.Parameter(torch.Tensor(self.out_features_per_partition, in_features))
        self.adapters_: MutableMapping[str, Adapter] = torch.nn.ModuleDict({})
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.adapters_) == 0:
            return F.linear(x, self.weight, self.bias)

        res = F.linear(x, self.weight, self.bias)

        adapter_func_list: List[Callable] = [
            self.__lora_forward,
        ]

        for func in adapter_func_list:
            res = func(data, res)

        return res

    def __lora_forward(
        self, data: torch.Tensor, result: torch.Tensor
    ) -> torch.Tensor:
        ## FIXME
        return result

    def load_adapter(self, adapter: Adapter):
        assert adapter.adapter_name_ not in self.adapters_
        self.adapters_[adapter.adapter_name_] = adapter

    def offload_adapter(self, adapter_name: str):
        if adapter_name not in self.adapters_:
            return

        del self.adapters_[adapter_name]

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
        res = F.linear(x, self.weight, self.bias)

        if len(self.adapters_) == 0:
            adapter_func_list: List[Callable] = [
                self.__lora_forward,
            ]
            
            for func in adapter_func_list:
                res = func(data, res)
        
        dist.all_reduce(res)
        return res

    def __lora_forward(
        self, data: torch.Tensor, result: torch.Tensor
    ) -> torch.Tensor:
        ## FIXME
        return result

    def load_adapter(self, adapter: Adapter):
        assert adapter.adapter_name_ not in self.adapters_
        self.adapters_[adapter.adapter_name_] = adapter

    def offload_adapter(self, adapter_name: str):
        if adapter_name not in self.adapters_:
            return

        del self.adapters_[adapter_name]