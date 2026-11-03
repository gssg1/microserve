import torch

from microserve.layers.fp8_chunk_prefill_attention import (
    fp8_chunk_prefill_attention,
)
from microserve.layers.fp8_decode_attention import fp8_decode_attention
from microserve.layers.fp8_kv_cache import store_fp8_kvcache


DEVICE = "cuda"
BLOCK_SIZE = 256
Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
SCALE = HEAD_DIM ** -0.5


def _alloc(num_blocks):
    shape = (num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_DIM)
    scale_shape = shape[:-1]
    k = torch.zeros(shape, device=DEVICE, dtype=torch.float8_e4m3fn)
    v = torch.zeros_like(k)
    ks = torch.ones(scale_shape, device=DEVICE, dtype=torch.float32)
    vs = torch.ones_like(ks)
    return k, v, ks, vs


def _gather(cache, scales, table, length):
    rows = []
    for position in range(length):
        block = int(table[position // BLOCK_SIZE])
        offset = position % BLOCK_SIZE
        rows.append(cache[block, offset].float() * scales[block, offset, :, None])
    return torch.stack(rows)


def test_decode():
    torch.manual_seed(1)
    lengths = [1, 73, 300]
    block_tables = torch.tensor([[0, 1], [2, 3], [4, 5]], device=DEVICE,
                                dtype=torch.int32)
    k_cache, v_cache, ks_cache, vs_cache = _alloc(6)
    keys, values, slots = [], [], []
    for batch, length in enumerate(lengths):
        keys.append(torch.randn(length, KV_HEADS, HEAD_DIM, device=DEVICE,
                                dtype=torch.float16) * 0.2)
        values.append(torch.randn(length, KV_HEADS, HEAD_DIM, device=DEVICE,
                                  dtype=torch.float16) * 0.2)
        for position in range(length):
            block = int(block_tables[batch, position // BLOCK_SIZE])
            slots.append(block * BLOCK_SIZE + position % BLOCK_SIZE)
    store_fp8_kvcache(
        torch.cat(keys), torch.cat(values), k_cache, v_cache, ks_cache,
        vs_cache, torch.tensor(slots, device=DEVICE, dtype=torch.int64))
    query = torch.randn(len(lengths), Q_HEADS, HEAD_DIM, device=DEVICE,
                        dtype=torch.float16) * 0.2
    actual = fp8_decode_attention(
        query, k_cache, v_cache, ks_cache, vs_cache,
        torch.tensor(lengths, device=DEVICE, dtype=torch.int32),
        block_tables, SCALE)
    head_map = torch.arange(Q_HEADS, device=DEVICE) // (Q_HEADS // KV_HEADS)
    refs = []
    for batch, length in enumerate(lengths):
        k = _gather(k_cache, ks_cache, block_tables[batch], length)[:, head_map]
        v = _gather(v_cache, vs_cache, block_tables[batch], length)[:, head_map]
        scores = torch.einsum("hd,lhd->hl", query[batch].float(), k) * SCALE
        refs.append(torch.einsum("hl,lhd->hd", scores.softmax(-1), v))
    reference = torch.stack(refs).half()
    error = (actual - reference).abs().max().item()
    assert torch.isfinite(actual).all()
    assert error < 0.004, error


def test_chunk_prefill():
    torch.manual_seed(2)
    prefix_len, query_len = 273, 31
    total = prefix_len + query_len
    table = torch.tensor([[1, 0]], device=DEVICE, dtype=torch.int32)
    k_cache, v_cache, ks_cache, vs_cache = _alloc(2)
    key = torch.randn(total, KV_HEADS, HEAD_DIM, device=DEVICE,
                      dtype=torch.float16) * 0.2
    value = torch.randn_like(key)
    slots = []
    for position in range(total):
        block = int(table[0, position // BLOCK_SIZE])
        slots.append(block * BLOCK_SIZE + position % BLOCK_SIZE)
    store_fp8_kvcache(
        key, value, k_cache, v_cache, ks_cache, vs_cache,
        torch.tensor(slots, device=DEVICE, dtype=torch.int64))
    query = torch.randn(query_len, Q_HEADS, HEAD_DIM, device=DEVICE,
                        dtype=torch.float16) * 0.2
    actual = fp8_chunk_prefill_attention(
        query, k_cache, v_cache, ks_cache, vs_cache, table, prefix_len, SCALE)
    head_map = torch.arange(Q_HEADS, device=DEVICE) // (Q_HEADS // KV_HEADS)
    k = _gather(k_cache, ks_cache, table[0], total)[:, head_map]
    v = _gather(v_cache, vs_cache, table[0], total)[:, head_map]
    refs = []
    for query_id in range(query_len):
        visible = prefix_len + query_id + 1
        scores = torch.einsum(
            "hd,lhd->hl", query[query_id].float(), k[:visible]) * SCALE
        refs.append(torch.einsum("hl,lhd->hd", scores.softmax(-1), v[:visible]))
    reference = torch.stack(refs).half()
    error = (actual - reference).abs().max().item()
    assert torch.isfinite(actual).all()
    assert error < 0.006, error


if __name__ == "__main__":
    test_decode()
    test_chunk_prefill()
    print("PASS: FP8 KV decode and chunk-prefill attention")
