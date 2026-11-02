import torch
import torch.nn.functional as F
from torch import nn

class VocabParallelEmbedding(nn.Module):
    """单卡：普通 embedding；命名保留以对齐 loader 的 get_parameter。"""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)

class ParallelLMHead(nn.Module):
    """语言模型头；行选择（每 seq 末位 logits）由 runner 在调用前完成。"""

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        super().__init__()
        assert not bias
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)
