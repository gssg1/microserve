import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_chunk_prefill_attention(
    Q, K_CACHE, V_CACHE, K_SCALE_CACHE, V_SCALE_CACHE, OUT, BLOCK_TABLE,
    PREFIX_LEN, QUERY_LEN, HEAD_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
):
    query_start = tl.program_id(0) * BLOCK_M
    q_head_id = tl.program_id(1)
    kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)
    q_ids = query_start + tl.arange(0, BLOCK_M)
    d = tl.arange(0, HEAD_DIM)
    q_mask = q_ids < QUERY_LEN
    q_ptrs = ((q_ids[:, None] * NUM_Q_HEADS + q_head_id) * HEAD_DIM
              + d[None, :])
    q = tl.load(Q + q_ptrs, mask=q_mask[:, None], other=0.0)
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    n_offsets = tl.arange(0, BLOCK_N)
    total_len = PREFIX_LEN + QUERY_LEN

    for start_n in range(0, total_len, BLOCK_N):
        logical_pos = start_n + n_offsets
        page_idx = start_n // BLOCK_SIZE
        page_offset = start_n % BLOCK_SIZE + n_offsets
        physical_block = tl.load(BLOCK_TABLE + page_idx)
        n_mask = logical_pos < total_len
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
        k = (k_fp8 * k_scale[:, None]).to(tl.float16)
        v = (v_fp8 * v_scale[:, None]).to(tl.float16)
        scores = tl.dot(q, tl.trans(k)) * SOFTMAX_SCALE
        visible_len = PREFIX_LEN + q_ids + 1
        causal = logical_pos[None, :] < visible_len[:, None]
        valid = q_mask[:, None] & n_mask[None, :] & causal
        scores = tl.where(valid, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(q_mask[:, None], p, 0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = acc / l_i[:, None]
    out_ptrs = ((q_ids[:, None] * NUM_Q_HEADS + q_head_id) * HEAD_DIM
                + d[None, :])
    tl.store(OUT + out_ptrs, out, mask=q_mask[:, None])


def fp8_chunk_prefill_attention(
    query: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor, v_scale_cache: torch.Tensor,
    block_table: torch.Tensor, prefix_len: int, softmax_scale: float,
    block_m: int | None = None, block_n: int | None = None,
) -> torch.Tensor:
    query = query.contiguous()
    block_table = block_table.contiguous().view(-1)
    query_len, num_q_heads, head_dim = query.shape
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    if block_m is None:
        block_m = 16 if query_len <= 32 else 32
    if block_n is None:
        if query_len <= 32:
            block_n = 128
        elif query_len <= 64:
            block_n = 64
        elif query_len <= 128:
            block_n = 32
        else:
            block_n = 64
    assert query.dtype == torch.float16
    assert k_cache.dtype == v_cache.dtype == torch.float8_e4m3fn
    assert k_scale_cache.dtype == v_scale_cache.dtype == torch.float32
    assert num_q_heads % num_kv_heads == 0
    assert head_dim == k_cache.shape[3]
    assert block_size % block_n == 0
    output = torch.empty_like(query)
    grid = (triton.cdiv(query_len, block_m), num_q_heads)
    _fp8_chunk_prefill_attention[grid](
        query, k_cache, v_cache, k_scale_cache, v_scale_cache, output,
        block_table, prefix_len, query_len, HEAD_DIM=head_dim,
        NUM_Q_HEADS=num_q_heads, NUM_KV_HEADS=num_kv_heads,
        BLOCK_SIZE=block_size, BLOCK_M=block_m, BLOCK_N=block_n,
        SOFTMAX_SCALE=softmax_scale, num_warps=4)
    return output
