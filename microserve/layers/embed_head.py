import torch
import torch.nn.functional as F
from torch import nn

from microserve.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from microserve.layers.linear import divide
from microserve.utils.profiling import profile_range


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_embeddings_per_partition = divide(num_embeddings, self.tp_size)
        self.vocab_start = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end = self.vocab_start + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(
            self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard = loaded_weight.narrow(
            0, self.vocab_start, self.num_embeddings_per_partition)
        param.data.copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.embedding(x, self.weight)
        mask = (x < self.vocab_start) | (x >= self.vocab_end)
        local_ids = (x - self.vocab_start).masked_fill(mask, 0)
        output = F.embedding(local_ids, self.weight)
        output = output.masked_fill(mask.unsqueeze(-1), 0)
        with profile_range("embedding.all_reduce", detailed=True):
            return tensor_model_parallel_all_reduce(output)


class ParallelLMHead(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        super().__init__()
        assert not bias
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_embeddings_per_partition = divide(num_embeddings, self.tp_size)
        self.vocab_start = self.tp_rank * self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(
            self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard = loaded_weight.narrow(
            0, self.vocab_start, self.num_embeddings_per_partition)
        param.data.copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)
