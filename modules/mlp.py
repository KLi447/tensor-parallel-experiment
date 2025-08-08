import torch
from torch import nn
import torch.nn.functional as F
from .linear import ColumnParallelLinear, RowParallelLinear
from .lora import LoraAdapter
from typing import Dict

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

    @property
    def linear_dict(self) -> Dict[str, nn.Module]:
        return {
            f"layers.{self.layer_id_}.mlp.gate_proj": self.gate_,
            f"layers.{self.layer_id_}.mlp.down_proj": self.down_,
            f"layers.{self.layer_id_}.mlp.up_proj": self.up_,
        }

    def load_adapter(self, adapter_name: str, r: int, lora_alpha: int, lora_dropout: float):
        self.gate_.load_adapter(adapter_name, r, lora_alpha, lora_dropout)
        self.up_.load_adapter(adapter_name, r, lora_alpha, lora_dropout)
        self.down_.load_adapter(adapter_name, r, lora_alpha, lora_dropout)

    def unload_adapter(self, adapter_name: str):
        self.gate_.unload_adapter(adapter_name)
        self.up_.unload_adapter(adapter_name)
        self.down_.unload_adapter(adapter_name)