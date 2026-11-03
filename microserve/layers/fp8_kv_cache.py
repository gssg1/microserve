import torch
import triton
import triton.language as tl


FP8_MAX = 448.0


@triton.jit
def _store_fp8_kvcache_kernel(
    K, V, K_CACHE, V_CACHE, K_SCALE_CACHE, V_SCALE_CACHE, SLOT_MAPPING,
    HEAD_DIM: tl.constexpr, NUM_KV_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < HEAD_DIM
    src = (token_id * NUM_KV_HEADS + kv_head_id) * HEAD_DIM + d
    k = tl.load(K + src, mask=d_mask, other=0.0).to(tl.float32)
    v = tl.load(V + src, mask=d_mask, other=0.0).to(tl.float32)
    k_scale = tl.maximum(tl.max(tl.abs(k), axis=0), 1e-8) / 448.0
    v_scale = tl.maximum(tl.max(tl.abs(v), axis=0), 1e-8) / 448.0
    k_fp8 = tl.clamp(k / k_scale, -448.0, 448.0).to(tl.float8e4nv)
    v_fp8 = tl.clamp(v / v_scale, -448.0, 448.0).to(tl.float8e4nv)

    slot = tl.load(SLOT_MAPPING + token_id)
    physical_block = slot // BLOCK_SIZE
    block_offset = slot % BLOCK_SIZE
    dst = (((physical_block * BLOCK_SIZE + block_offset) * NUM_KV_HEADS
            + kv_head_id) * HEAD_DIM + d)
    tl.store(K_CACHE + dst, k_fp8, mask=d_mask)
    tl.store(V_CACHE + dst, v_fp8, mask=d_mask)
    scale_dst = ((physical_block * BLOCK_SIZE + block_offset)
                 * NUM_KV_HEADS + kv_head_id)
    tl.store(K_SCALE_CACHE + scale_dst, k_scale)
    tl.store(V_SCALE_CACHE + scale_dst, v_scale)


def store_fp8_kvcache(
    key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor,
    v_cache: torch.Tensor, k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor, slot_mapping: torch.Tensor,
) -> None:
    key = key.contiguous()
    value = value.contiguous()
    slot_mapping = slot_mapping.contiguous()
    num_tokens, num_kv_heads, head_dim = key.shape
    block_size = k_cache.shape[1]
    assert value.shape == key.shape
    assert k_cache.dtype == v_cache.dtype == torch.float8_e4m3fn
    assert k_scale_cache.dtype == v_scale_cache.dtype == torch.float32
    assert k_cache.shape[2:] == (num_kv_heads, head_dim)
    assert k_scale_cache.shape == k_cache.shape[:-1]
    _store_fp8_kvcache_kernel[(num_tokens, num_kv_heads)](
        key, value, k_cache, v_cache, k_scale_cache, v_scale_cache,
        slot_mapping, HEAD_DIM=head_dim, NUM_KV_HEADS=num_kv_heads,
        BLOCK_SIZE=block_size, BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4)
