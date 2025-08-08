import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import MutableMapping, List, Callable
from .lora import LoraAdapter


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, world_size: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.out_features_per_partition = out_features // world_size

        self.weight = nn.Parameter(torch.Tensor(self.out_features_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)

        self.adapters: MutableMapping[str, LoraAdapter] = torch.nn.ModuleDict({})

    def __lora_forward(self, x: torch.Tensor, base_result: torch.Tensor) -> torch.Tensor:
        lora_result = base_result
        for adapter in self.adapters.values():
            lora_output = adapter(x)
            lora_result = lora_result + lora_output
        return lora_result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = F.linear(x, self.weight, self.bias)

        if len(self.adapters) > 0:
            res = self.__lora_forward(x, res)
            
        return res

    def load_adapter(self, adapter_name: str, r: int, lora_alpha: int, lora_dropout: float):
        if adapter_name in self.adapters:
            print(f"Adapter '{adapter_name}' already exists. Overwriting.")
        
        adapter = LoraAdapter(
            adapter_name, self.in_features, self.out_features_per_partition, r, lora_alpha, lora_dropout
        )
        adapter.to(self.weight.device)
        self.adapters[adapter_name] = adapter

    def unload_adapter(self, adapter_name: str):
        if adapter_name in self.adapters:
            del self.adapters[adapter_name]
        else:
            print(f"Warning: Adapter '{adapter_name}' not found.")


class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, world_size: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.in_features_per_partition = in_features // world_size
        
        self.weight = nn.Parameter(torch.Tensor(out_features, self.in_features_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.adapters: MutableMapping[str, LoraAdapter] = torch.nn.ModuleDict({})

    def __lora_forward(self, x: torch.Tensor, base_result: torch.Tensor) -> torch.Tensor:
        lora_result = base_result
        for adapter in self.adapters.values():
            lora_output = adapter(x)
            lora_result = lora_result + lora_output
        return lora_result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = F.linear(x, self.weight)

        if len(self.adapters) > 0:
            res = self.__lora_forward(x, res)
        
        if self.world_size > 1:
            dist.all_reduce(res)

        if self.bias is not None:
            res = res + self.bias
            
        return res

    def load_adapter(self, adapter_name: str, r: int, lora_alpha: int, lora_dropout: float):
        if adapter_name in self.adapters:
            print(f"Adapter '{adapter_name}' already exists. Overwriting.")
        
        adapter = LoraAdapter(
            adapter_name, self.in_features_per_partition, self.out_features, r, lora_alpha, lora_dropout
        )
        adapter.to(self.weight.device)
        self.adapters[adapter_name] = adapter

    def unload_adapter(self, adapter_name: str):
        if adapter_name in self.adapters:
            del self.adapters[adapter_name]
        else:
            print(f"Warning: Adapter '{adapter_name}' not found.")

