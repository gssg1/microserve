import torch
from torch import nn

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from microserve.layers.fp8_chunk_prefill_attention import fp8_chunk_prefill_attention
from microserve.layers.fp8_decode_attention import fp8_decode_attention
from microserve.layers.fp8_kv_cache import store_fp8_kvcache
from microserve.utils.context import AttentionMode, get_context

def store_kvcache(key: torch.Tensor, value: torch.Tensor,
                  k_cache: torch.Tensor, v_cache: torch.Tensor,
                  slot_mapping: torch.Tensor):

    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    key = key.reshape(N, D).contiguous()
    value = value.reshape(N, D).contiguous()
    sm = slot_mapping if slot_mapping.dtype == torch.int64 else slot_mapping.long()
    k_cache.reshape(-1, D).index_copy_(0, sm, key)
    v_cache.reshape(-1, D).index_copy_(0, sm, value)

class Attention(nn.Module):
    """paged attention。k_cache/v_cache 由 model_runner 注入（普通属性，不进 state_dict）。"""

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale_cache = self.v_scale_cache = None
        self.kv_cache_dtype = "fp16"

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        assert k_cache.numel(), "k_cache 未注入：model_runner 须先 allocate_kv_cache"
        if self.kv_cache_dtype == "fp8_e4m3":
            store_fp8_kvcache(
                k, v, k_cache, v_cache, self.k_scale_cache,
                self.v_scale_cache, ctx.slot_mapping)
            if ctx.mode is AttentionMode.DECODE:
                return fp8_decode_attention(
                    q, k_cache, v_cache, self.k_scale_cache,
                    self.v_scale_cache, ctx.cache_seqlens,
                    ctx.block_tables, self.scale)
            if ctx.mode is AttentionMode.PREFILL:
                return flash_attn_varlen_func(
                    q, k, v, cu_seqlens_q=ctx.cu_seqlens_q,
                    cu_seqlens_k=ctx.cu_seqlens_k,
                    max_seqlen_q=ctx.max_seqlen_q,
                    max_seqlen_k=ctx.max_seqlen_k,
                    softmax_scale=self.scale, causal=True)
            if ctx.mode is AttentionMode.CHUNK_PREFILL:
                return fp8_chunk_prefill_attention(
                    q, k_cache, v_cache, self.k_scale_cache,
                    self.v_scale_cache, ctx.block_tables,
                    ctx.prefix_len, self.scale)
            raise RuntimeError(f"unsupported attention mode: {ctx.mode}")

        store_kvcache(k, v, k_cache, v_cache, ctx.slot_mapping)
        if ctx.mode is AttentionMode.DECODE:
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=ctx.cache_seqlens, block_table=ctx.block_tables,
                softmax_scale=self.scale, causal=True)
            return o.squeeze(1)
        if ctx.mode is AttentionMode.PREFILL:
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=ctx.cu_seqlens_q,
                cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                softmax_scale=self.scale, causal=True)
        if ctx.mode is AttentionMode.CHUNK_PREFILL:
            o = flash_attn_varlen_func(
                q, k_cache, v_cache,
                cu_seqlens_q=ctx.cu_seqlens_q, cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q, max_seqlen_k=ctx.max_seqlen_k,
                softmax_scale=self.scale, causal=True, block_table=ctx.block_tables)
            return o
        raise RuntimeError(f"unsupported attention mode: {ctx.mode}")
