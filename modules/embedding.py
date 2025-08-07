import torch
from torch import nn
import torch.distributed as dist

class ParallelEmbedding(nn.Module):
    def __init__(self, config, world_size: int):
        super().__init__()
        vocab_size_per_partition = config.vocab_size // world_size
        self.embedding = nn.Embedding(
            vocab_size_per_partition,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.vocab_start_index = dist.get_rank() * vocab_size_per_partition
        self.vocab_end_index = self.vocab_start_index + vocab_size_per_partition

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens < self.vocab_start_index) | (tokens >= self.vocab_end_index)
        masked_tokens = tokens.clone() - self.vocab_start_index
        masked_tokens[mask] = self.embedding.padding_idx if self.embedding.padding_idx is not None else 0

        partial_embeddings = self.embedding(masked_tokens)

        partial_embeddings[mask.unsqueeze(-1).expand_as(partial_embeddings)] = 0.0

        dist.all_reduce(partial_embeddings)
        return partial_embeddings