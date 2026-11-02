import torch
from torch import nn

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """transformers 同款：x = [x1, x2] -> [-x2, x1]。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_size: int, rotary_dim: int,
                 max_position_embeddings: int, base: float) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze_(1)
        sin = emb.sin().unsqueeze_(1)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor,
                key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = query.dtype
        cos = self.cos_cache[positions].to(dtype)
        sin = self.sin_cache[positions].to(dtype)
        q_embed = (query * cos) + (rotate_half(query) * sin)
        k_embed = (key * cos) + (rotate_half(key) * sin)
        return q_embed, k_embed

_ROPE_CACHE: dict = {}

def get_rope(head_size: int, rotary_dim: int, max_position: int,
             base: float) -> RotaryEmbedding:
    """每 (head_size, base) 只建一次（cos/sin 表占内存）。"""
    key = (head_size, rotary_dim, max_position, base)
    if key not in _ROPE_CACHE:
        _ROPE_CACHE[key] = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return _ROPE_CACHE[key]
