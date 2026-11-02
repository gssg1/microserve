from dataclasses import dataclass
from enum import Enum, auto

import torch

_CTX = None

class AttentionMode(Enum):
    DECODE = auto()
    PREFILL = auto()
    CHUNK_PREFILL = auto()

@dataclass(slots=True)
class Context:
    mode: AttentionMode = AttentionMode.PREFILL

    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    cache_seqlens: torch.Tensor | None = None

    prefix_len: int = 0
    query_len: int = 0

    slot_mapping: torch.Tensor | None = None

    block_tables: torch.Tensor | None = None

def get_context() -> Context:
    global _CTX
    if _CTX is None:
        _CTX = Context()
    return _CTX

def set_context(ctx: Context):
    global _CTX
    _CTX = ctx

def reset_context():
    global _CTX
    _CTX = Context()
