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

    @property
    def linear_dict(self) -> Dict[str, Linear]:
        return {
            f"layers.{self.layer_id_}.mlp.gate_proj": self.gate_,
            f"layers.{self.layer_id_}.mlp.down_proj": self.down_,
            f"layers.{self.layer_id_}.mlp.up_proj": self.up_,
        }

    def load_adapter(self, adapter_model: AdapterModel):
        for name, module in self.linear_dict.items():
            if name not in adapter_model:
                continue
            module.load_adapter(adapter_model[name])

    def offload_adapter(self, adapter_name: str):
        for _, module in self.linear_dict.items():
            module.offload_adapter(adapter_name)