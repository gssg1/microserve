from pathlib import Path

import torch
from transformers import AutoConfig

from microserve.Config import Config
from microserve.distributed import (broadcast_tensor, gather_tensor, get_device,
                                    get_tensor_model_parallel_world_size,
                                    is_rank_zero, tensor_model_parallel_min)
from microserve.engine.sequence import Sequence
from microserve.layers.attention import Attention
from microserve.layers.fp8_linear import quantize_fp8_linears
from microserve.models.qwen3 import Qwen3ForCausalLM
from microserve.utils.context import (AttentionMode, Context, get_context,
                                      reset_context, set_context)
from microserve.utils.loader import load_model
from microserve.utils.profiling import (add_metric_value, cpu_profile,
                                        profile_range)

def _sample(logits: torch.Tensor, seq: Sequence) -> int:
    p = seq.sampling_params
    logits = logits.float()
    if p.temperature is not None and p.temperature > 0:
        logits = logits / p.temperature
        if p.top_k is not None and p.top_k > 0:
            k = min(p.top_k, logits.numel())
            topk = torch.topk(logits, k).values[-1]
            logits = logits.masked_fill(logits < topk, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        if p.top_p is not None and 0.0 < p.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumsum - sorted_probs > p.top_p
            sorted_probs.masked_fill_(cutoff, 0.0)
            probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return int(torch.multinomial(probs, 1).item())
    return int(torch.argmax(logits).item())

class ModelRunner:
    def __init__(self, config: Config):
        self.config = config
        self.block_size = config.block_size
        self.num_blocks = config.num_kvcache_blocks
        self.tp_size = get_tensor_model_parallel_world_size()
        if config.tensor_parallel_size != self.tp_size:
            raise ValueError(
                f"tensor_parallel_size={config.tensor_parallel_size}, "
                f"WORLD_SIZE={self.tp_size}")
        self.enforce_eager = config.enforce_eager or self.tp_size > 1
        self.device = get_device()

        self.hf_config = AutoConfig.from_pretrained(config.model_dir,
                                                    trust_remote_code=True)
        dtype = config.dtype
        assert self.hf_config.num_attention_heads % self.tp_size == 0
        assert self.hf_config.num_key_value_heads % self.tp_size == 0
        assert self.hf_config.intermediate_size % self.tp_size == 0
        assert self.hf_config.vocab_size % self.tp_size == 0
        self.num_kv_heads = self.hf_config.num_key_value_heads // self.tp_size
        self.head_dim = getattr(self.hf_config, "head_dim", None) or (
            self.hf_config.hidden_size // self.hf_config.num_attention_heads)
        self.num_layers = self.hf_config.num_hidden_layers
        self.max_model_len = min(config.max_model_len,
                                 self.hf_config.max_position_embeddings)
        self.vocab_size = self.hf_config.vocab_size

        if config.seed is not None:
            torch.manual_seed(config.seed)

        self.model = Qwen3ForCausalLM(
            self.hf_config, fused_rmsnorm=config.fused_rmsnorm).to(dtype)
        load_model(self.model, config.model_dir)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.num_quantized_linears = 0
        if config.quantization == "fp8_dynamic":
            if dtype != torch.float16:
                raise ValueError("FP8 quantization requires torch.float16")
            self.num_quantized_linears = quantize_fp8_linears(self.model)

        if self.num_blocks <= 0:
            self.num_blocks = tensor_model_parallel_min(self._auto_num_blocks())
        else:
            self.num_blocks = tensor_model_parallel_min(self.num_blocks)

        self.garbage_slot = self.num_blocks * self.block_size
        cache_dtype = (torch.float8_e4m3fn
                       if config.kv_cache_dtype == "fp8_e4m3" else dtype)
        self.kv_cache = torch.zeros(
            2, self.num_layers, self.num_blocks + 1, self.block_size,
            self.num_kv_heads, self.head_dim, dtype=cache_dtype, device=self.device)
        self.kv_scale_cache = None
        if config.kv_cache_dtype == "fp8_e4m3":
            self.kv_scale_cache = torch.ones(
                2, self.num_layers, self.num_blocks + 1, self.block_size,
                self.num_kv_heads, dtype=torch.float32, device=self.device)
        self._inject_kv_cache()

        self.graph_bs: list[int] = []
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_pool: object | None = None
        self.graph_vars: dict | None = None
        if not self.enforce_eager:
            self._warmup()
            self._capture_decode_graphs()

    def _auto_num_blocks(self) -> int:
        cfg = self.config
        free, total = torch.cuda.mem_get_info()
        if cfg.kv_cache_dtype == "fp8_e4m3":
            data_bytes = (2 * self.num_layers * self.block_size
                          * self.num_kv_heads * self.head_dim)
            scale_bytes = (2 * self.num_layers * self.block_size
                           * self.num_kv_heads * 4)
            bytes_per_block = data_bytes + scale_bytes
        else:
            bytes_per_block = (2 * self.num_layers * self.block_size
                               * self.num_kv_heads * self.head_dim
                               * torch.tensor([], dtype=cfg.dtype).element_size())

        available = (free + total * (cfg.gpu_memory_utilization - 1.0)) * 0.8
        return max(1, int(available // bytes_per_block))

    def _inject_kv_cache(self):
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                if self.kv_scale_cache is not None:
                    module.k_scale_cache = self.kv_scale_cache[0, layer_id]
                    module.v_scale_cache = self.kv_scale_cache[1, layer_id]
                module.kv_cache_dtype = self.config.kv_cache_dtype
                layer_id += 1
        assert layer_id == self.num_layers

    def _compile_decode(self, seqs: list[Sequence]):
        bs = len(seqs)
        input_ids = torch.empty(bs, dtype=torch.int64, device=self.device)
        positions = torch.empty(bs, dtype=torch.int64, device=self.device)
        slot = torch.empty(bs, dtype=torch.int64, device=self.device)
        cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
        for i, seq in enumerate(seqs):
            p = seq.num_computed_tokens
            assert seq.gap == 1 and seq.num_scheduled_tokens == 1
            input_ids[i] = seq.token_ids[p]
            positions[i] = p
            slot[i] = (seq.block_table[p // self.block_size] * self.block_size
                       + p % self.block_size)
            cache_seqlens[i] = p + 1
        ctx = Context(
            mode=AttentionMode.DECODE, slot_mapping=slot,
            cache_seqlens=cache_seqlens,
            block_tables=self._make_block_tables(seqs))
        return ctx, input_ids, positions

    def _compile_prefill(self, seqs: list[Sequence]):
        bs = len(seqs)
        total_q = sum(s.num_scheduled_tokens for s in seqs)
        input_ids = torch.empty(total_q, dtype=torch.int64, device=self.device)
        positions = torch.empty(total_q, dtype=torch.int64, device=self.device)
        slot = torch.empty(total_q, dtype=torch.int64, device=self.device)
        cu_q = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        max_q = 0
        off = 0
        for i, seq in enumerate(seqs):
            assert seq.num_computed_tokens == 0 and seq.gap > 1
            n = seq.num_scheduled_tokens
            for j in range(n):
                p = j
                input_ids[off + j] = seq.token_ids[p]
                positions[off + j] = p
                slot[off + j] = (seq.block_table[p // self.block_size]
                                 * self.block_size + p % self.block_size)
            cu_q[i + 1] = cu_q[i] + n
            max_q = max(max_q, n)
            off += n
        ctx = Context(
            mode=AttentionMode.PREFILL, cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_q, max_seqlen_q=max_q, max_seqlen_k=max_q,
            slot_mapping=slot)
        return ctx, cu_q, input_ids, positions

    def _compile_chunk_prefill(self, seq: Sequence):
        start = seq.num_computed_tokens
        n = seq.num_scheduled_tokens
        assert start > 0 and seq.gap > 1 and n > 0
        input_ids = torch.empty(n, dtype=torch.int64, device=self.device)
        positions = torch.empty(n, dtype=torch.int64, device=self.device)
        slot = torch.empty(n, dtype=torch.int64, device=self.device)
        for j in range(n):
            p = start + j
            input_ids[j] = seq.token_ids[p]
            positions[j] = p
            slot[j] = (seq.block_table[p // self.block_size] * self.block_size
                       + p % self.block_size)
        cu_q = torch.tensor([0, n], dtype=torch.int32, device=self.device)
        cu_k = torch.tensor([0, start + n], dtype=torch.int32, device=self.device)
        ctx = Context(
            mode=AttentionMode.CHUNK_PREFILL, slot_mapping=slot,
            block_tables=self._make_block_tables([seq]),
            cache_seqlens=torch.tensor([start + n], dtype=torch.int32,
                                        device=self.device),
            prefix_len=start, query_len=n, cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k, max_seqlen_q=n, max_seqlen_k=start + n)
        return ctx, input_ids, positions

    @staticmethod
    def _partition_batch(seqs: list[Sequence]):
        decode, prefill, chunk = [], [], []
        for index, seq in enumerate(seqs):
            item = (index, seq)
            if seq.gap == 1:
                decode.append(item)
            elif seq.num_computed_tokens == 0:
                prefill.append(item)
            else:
                chunk.append(item)
        return decode, prefill, chunk

    def _make_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        max_len = max(len(s.block_table) for s in seqs)
        bt = torch.full((len(seqs), max_len), -1, dtype=torch.int32, device=self.device)
        for i, s in enumerate(seqs):
            if s.block_table:
                bt[i, :len(s.block_table)] = torch.tensor(s.block_table,
                                                          dtype=torch.int32,
                                                          device=self.device)
        return bt

    @torch.inference_mode()
    def execute(self, seqs: list[Sequence]) -> list[int | None]:
        with cpu_profile("runner.partition_batch"):
            decode, prefill, chunk = self._partition_batch(seqs)
        if chunk and not self.config.enable_chunked_prefill:
            raise RuntimeError("chunk-prefill batch produced while chunked prefill is disabled")
        add_metric_value(
            "first_prefill_tokens",
            sum(seq.num_scheduled_tokens for _, seq in prefill),
        )
        add_metric_value(
            "chunk_prefill_tokens",
            sum(seq.num_scheduled_tokens for _, seq in chunk),
        )
        outputs: list[int | None] = [None] * len(seqs)
        try:
            if decode:
                indices, group = zip(*decode)
                group = list(group)
                with cpu_profile("decode.compile"):
                    ctx, input_ids, positions = self._compile_decode(group)
                set_context(ctx)
                with profile_range("decode.model", measure_cuda=True):
                    hidden = self._run_decode(input_ids, positions)
                with profile_range("decode.lm_head", measure_cuda=True):
                    local_logits = self.model.compute_logits(hidden)
                with profile_range("decode.logits_all_gather", measure_cuda=True):
                    logits = self._gather_logits(local_logits)
                sampled = self._sample_rows(
                    group, logits, list(range(len(group))), "decode")
                for index, token in zip(indices, sampled):
                    outputs[index] = token

            if prefill:
                indices, group = zip(*prefill)
                group = list(group)
                with cpu_profile("prefill.compile"):
                    ctx, cu_q, input_ids, positions = self._compile_prefill(group)
                set_context(ctx)
                with profile_range("prefill.model", measure_cuda=True):
                    hidden = self.model(input_ids, positions)
                last_idx = (cu_q[1:] - 1)
                hidden = hidden[last_idx]
                with profile_range("prefill.lm_head", measure_cuda=True):
                    local_logits = self.model.compute_logits(hidden)
                with profile_range("prefill.logits_all_gather", measure_cuda=True):
                    logits = self._gather_logits(local_logits)
                sampled = self._sample_rows(
                    group, logits, list(range(len(group))), "prefill")
                for index, token in zip(indices, sampled):
                    outputs[index] = token

            if self.config.enable_chunked_prefill:
                for index, seq in chunk:
                    with cpu_profile("chunk_prefill.compile"):
                        ctx, input_ids, positions = self._compile_chunk_prefill(seq)
                    set_context(ctx)
                    with profile_range("chunk_prefill.model", measure_cuda=True):
                        hidden = self.model(input_ids, positions)
                    with profile_range("chunk_prefill.lm_head", measure_cuda=True):
                        local_logits = self.model.compute_logits(hidden[-1:])
                    with profile_range(
                            "chunk_prefill.logits_all_gather", measure_cuda=True):
                        logits = self._gather_logits(local_logits)
                    outputs[index] = self._sample_rows(
                        [seq], logits, [0], "chunk_prefill")[0]
        finally:
            reset_context()
        return outputs

    def _run_decode(self, input_ids: torch.Tensor,
                    positions: torch.Tensor) -> torch.Tensor:
        bs = input_ids.size(0)
        if not self.enforce_eager and self.graphs and bs <= self.graph_bs[-1]:
            gv = self.graph_vars
            bs_ = next(b for b in self.graph_bs if b >= bs)
            graph = self.graphs[bs_]
            gv["input_ids"][:bs] = input_ids
            gv["positions"][:bs] = positions
            ctx = get_context()

            gv["slot_mapping"].fill_(self.garbage_slot)
            gv["slot_mapping"][:bs] = ctx.slot_mapping
            gv["cache_seqlens"].zero_()
            gv["cache_seqlens"][:bs] = ctx.cache_seqlens
            gv["block_tables"].fill_(-1)
            gv["block_tables"][:bs, :ctx.block_tables.size(1)] = ctx.block_tables
            set_context(Context(mode=AttentionMode.DECODE,
                                slot_mapping=gv["slot_mapping"][:bs_],
                                cache_seqlens=gv["cache_seqlens"][:bs_],
                                block_tables=gv["block_tables"][:bs_]))
            graph.replay()
            set_context(ctx)
            return gv["outputs"][:bs].clone()
        return self.model(input_ids, positions)

    def _sample_rows(self, seqs: list[Sequence], logits: torch.Tensor,
                     rows: list[int], mode: str) -> list[int | None]:
        out: list[int | None] = []
        for i, seq in enumerate(seqs):
            if not seq.needs_sample:
                out.append(None)
            else:
                token = torch.empty(1, dtype=torch.int64, device=self.device)
                if is_rank_zero():
                    with profile_range(f"{mode}.sample", measure_cuda=True):
                        token[0] = _sample(logits[i], seq)
                with profile_range(
                        f"{mode}.token_broadcast", measure_cuda=True):
                    broadcast_tensor(token)
                out.append(int(token.item()))
        return out

    def _gather_logits(self, local_logits: torch.Tensor) -> torch.Tensor:
        return gather_tensor(local_logits, dim=-1)

    def _warmup(self):
        ctx = Context(
            mode=AttentionMode.DECODE,
            slot_mapping=torch.tensor([0], dtype=torch.int32, device=self.device),
            cache_seqlens=torch.tensor([1], dtype=torch.int32, device=self.device),
            block_tables=torch.tensor([[0]], dtype=torch.int32, device=self.device))
        set_context(ctx)
        try:
            self.model(torch.tensor([0], device=self.device),
                       torch.tensor([0], device=self.device))
        finally:
            reset_context()

    def _capture_decode_graphs(self):
        max_bs = self.config.max_num_seqs
        self.graph_bs = sorted({1, 2, 4, 8} | set(range(16, max_bs + 1, 16)))
        max_blocks = (self.max_model_len + self.block_size - 1) // self.block_size
        gv = dict(
            input_ids=torch.zeros(max_bs, dtype=torch.int64, device=self.device),
            positions=torch.zeros(max_bs, dtype=torch.int64, device=self.device),
            slot_mapping=torch.zeros(max_bs, dtype=torch.int64, device=self.device),
            cache_seqlens=torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            block_tables=torch.full((max_bs, max_blocks), -1, dtype=torch.int32,
                                    device=self.device),
            outputs=torch.zeros(max_bs, self.hf_config.hidden_size,
                                dtype=self.config.dtype, device=self.device),
        )
        self.graph_vars = gv
        for bs in self.graph_bs:
            gv["slot_mapping"][:bs].fill_(self.garbage_slot)
            ctx = Context(mode=AttentionMode.DECODE,
                          slot_mapping=gv["slot_mapping"][:bs],
                          cache_seqlens=gv["cache_seqlens"][:bs],
                          block_tables=gv["block_tables"][:bs])
            set_context(ctx)

            gv["outputs"][:bs] = self.model(gv["input_ids"][:bs], gv["positions"][:bs])
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.graph_pool):
                gv["outputs"][:bs] = self.model(gv["input_ids"][:bs], gv["positions"][:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            reset_context()
        torch.cuda.synchronize()
