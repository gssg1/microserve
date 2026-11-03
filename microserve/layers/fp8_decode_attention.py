import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_decode_flashAttention(
    Q, K_CACHE, V_CACHE, K_SCALE_CACHE, V_SCALE_CACHE, OUT, SEQ_LENS,
    BLOCK_TABLE, HEAD_DIM: tl.constexpr, NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr, SOFTMAX_SCALE: tl.constexpr,
    MAX_NUM_BLOCKS: tl.constexpr,
):
    seq_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)
    seq_len = tl.load(SEQ_LENS + seq_id)
    d = tl.arange(0, HEAD_DIM)
    q_ptr = (seq_id * NUM_Q_HEADS + q_head_id) * HEAD_DIM + d
    q = tl.load(Q + q_ptr).to(tl.float32)
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((HEAD_DIM,), tl.float32)
    n_offsets = tl.arange(0, BLOCK_N)

    for start_n in range(0, seq_len, BLOCK_N):
        logical_pos = start_n + n_offsets
        page_idx = start_n // BLOCK_SIZE
        page_offset = start_n % BLOCK_SIZE + n_offsets
        physical_block = tl.load(
            BLOCK_TABLE + seq_id * MAX_NUM_BLOCKS + page_idx)
        n_mask = logical_pos < seq_len
        cache_ptrs = (((physical_block * BLOCK_SIZE + page_offset[:, None])
                       * NUM_KV_HEADS + kv_head_id) * HEAD_DIM + d[None, :])
        k_fp8 = tl.load(K_CACHE + cache_ptrs, mask=n_mask[:, None],
                        other=0.0).to(tl.float32)
        v_fp8 = tl.load(V_CACHE + cache_ptrs, mask=n_mask[:, None],
                        other=0.0).to(tl.float32)
        scale_ptrs = ((physical_block * BLOCK_SIZE + page_offset)
                      * NUM_KV_HEADS + kv_head_id)
        k_scale = tl.load(K_SCALE_CACHE + scale_ptrs, mask=n_mask,
                          other=0.0).to(tl.float32)
        v_scale = tl.load(V_SCALE_CACHE + scale_ptrs, mask=n_mask,
                          other=0.0).to(tl.float32)
        scores = tl.sum(k_fp8 * q[None, :], axis=1)
        scores *= k_scale * SOFTMAX_SCALE
        scores = tl.where(n_mask, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        acc = (acc * alpha
               + tl.sum(p[:, None] * v_fp8 * v_scale[:, None], axis=0))
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    result = tl.where(seq_len > 0, acc / l_i, 0.0)
    tl.store(OUT + q_ptr, result)


def fp8_decode_attention(
    query: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor, v_scale_cache: torch.Tensor,
    cache_seqlens: torch.Tensor, block_tables: torch.Tensor,
    softmax_scale: float, block_n: int | None = None,
) -> torch.Tensor:
    query = query.contiguous()
    cache_seqlens = cache_seqlens.contiguous()
    block_tables = block_tables.contiguous()
    batch_size, num_q_heads, head_dim = query.shape
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    max_num_blocks = block_tables.shape[1]
    if block_n is None:
        if batch_size >= 64:
            block_n = 64
        elif batch_size >= 32 and max_num_blocks >= 4:
            block_n = 128
        else:
            block_n = 256
    assert query.dtype == torch.float16
    assert k_cache.dtype == v_cache.dtype == torch.float8_e4m3fn
    assert k_scale_cache.dtype == v_scale_cache.dtype == torch.float32
    assert cache_seqlens.dtype == torch.int32
    assert block_tables.dtype == torch.int32
    assert num_q_heads % num_kv_heads == 0
    assert head_dim == k_cache.shape[3]
    assert block_size % block_n == 0
    output = torch.empty_like(query)
    _fp8_decode_flashAttention[(batch_size, num_q_heads)](
        query, k_cache, v_cache, k_scale_cache, v_scale_cache, output,
        cache_seqlens, block_tables, HEAD_DIM=head_dim,
        NUM_Q_HEADS=num_q_heads, NUM_KV_HEADS=num_kv_heads,
        BLOCK_SIZE=block_size, BLOCK_N=block_n,
        SOFTMAX_SCALE=softmax_scale, MAX_NUM_BLOCKS=max_num_blocks,
        num_warps=4)
    return output
