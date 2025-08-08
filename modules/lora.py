import torch
from torch import nn
import torch.nn.functional as F

class LoraAdapter(nn.Module):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        super().__init__()
        self.adapter_name = adapter_name
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)

        if r > 0:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = 1.0

        self.lora_A = nn.Parameter(torch.Tensor(r, in_features))
        self.lora_B = nn.Parameter(torch.Tensor(out_features, r))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lora_dropout(x)
        x = F.linear(x, self.lora_A)
        x = F.linear(x, self.lora_B)
        return x * self.scaling