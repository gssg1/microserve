from time import perf_counter_ns

from microserve.Config import Config
from microserve.distributed import initialize_distributed
from microserve.engine.model_runner import ModelRunner
from microserve.engine.scheduler import Scheduler
from microserve.engine.sequence import SamplingParams, Sequence, SequenceStatus
from microserve.utils.metrics import MetricsCollector
from microserve.utils.profiling import configure_profiling, profile_range

class LLMEngine:
    def __init__(self, config: Config):
        initialize_distributed()
        self.config = config
        self.metrics = MetricsCollector(
            enabled=config.enable_metrics,
            cuda_events=config.profile_cuda_events,
        )
        configure_profiling(
            self.metrics,
            ranges_enabled=config.enable_nvtx,
            layers_enabled=config.profile_layers,
        )
        self.runner = ModelRunner(config)
        self.scheduler = Scheduler(
            num_kvcache_blocks=self.runner.num_blocks,
            block_size=config.block_size,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            policy=config.scheduler_policy,
            eos_token_id=config.eos_token_id,
            enable_chunked_prefill=config.enable_chunked_prefill)

    def add(self, token_ids: list[int],
            sampling_params: SamplingParams | None = None,
            max_tokens: int = 128) -> Sequence:
        if (not self.config.enable_chunked_prefill
                and len(token_ids) > self.config.max_num_batched_tokens):
            raise ValueError(
                "prompt length exceeds max_num_batched_tokens while chunked prefill is disabled")
        if (not self.config.enable_chunked_prefill
                and len(token_ids) > self.runner.num_blocks * self.config.block_size):
            raise ValueError(
                "prompt length exceeds KV cache capacity while chunked prefill is disabled")
        seq = Sequence(token_ids, sampling_params=sampling_params,
                       max_tokens=max_tokens)
        self.scheduler.add(seq)
        return seq

    def step(self) -> list[Sequence]:
        if not self.metrics.enabled and not self.config.enable_nvtx:
            batch = self.scheduler.schedule()
            if batch is None:
                return []
            outputs = self.runner.execute(batch)
            self.scheduler.postprocess(batch, outputs)
            return [s for s in batch if s.status == SequenceStatus.FINISHED]
        return self._profiled_step()

    def _profiled_step(self) -> list[Sequence]:
        manager = self.scheduler.block_manager
        before = {
            "waiting_before": len(self.scheduler.waiting),
            "running_before": len(self.scheduler.running),
            "free_blocks_before": manager.free_blocks_nums,
        }
        blocks_allocated = manager.total_blocks_allocated
        blocks_freed = manager.total_blocks_freed
        prefix_cache_hit_tokens = manager.total_prefix_cache_hit_tokens
        started = perf_counter_ns()
        with profile_range("scheduler.schedule"):
            batch = self.scheduler.schedule()
        schedule_ms = (perf_counter_ns() - started) / 1_000_000
        if batch is None:
            return []
        stats = self.scheduler.last_batch_stats or {}
        self.metrics.begin_step({
            **before,
            "batch_size": len(batch),
            "waiting_after_schedule": len(self.scheduler.waiting),
            "running_after_schedule": len(self.scheduler.running),
            "free_blocks_after_schedule": manager.free_blocks_nums,
            "allocated_blocks": manager.total_blocks_allocated - blocks_allocated,
            "freed_blocks": manager.total_blocks_freed - blocks_freed,
            "prefix_cache_hit_tokens": (
                manager.total_prefix_cache_hit_tokens
                - prefix_cache_hit_tokens),
            "scheduled_prefill_tokens": stats.get("prefill_tokens", 0),
            "scheduled_decode_tokens": stats.get("decode_tokens", 0),
            "has_prefill": stats.get("prefill", False),
            "has_decode": stats.get("decode", False),
        })
        self.metrics.add_cpu("scheduler.schedule", schedule_ms)
        try:
            with profile_range("runner.execute", measure_cuda=True):
                outputs = self.runner.execute(batch)
            started = perf_counter_ns()
            with profile_range("scheduler.postprocess"):
                self.scheduler.postprocess(batch, outputs)
            self.metrics.add_cpu(
                "scheduler.postprocess",
                (perf_counter_ns() - started) / 1_000_000,
            )
        finally:
            self.metrics.end_step()
        return [s for s in batch if s.status == SequenceStatus.FINISHED]

    def is_idle(self) -> bool:
        return self.scheduler.is_idle()

    def generate(self, prompt_token_ids: list[int],
                 sampling_params: SamplingParams | None = None,
                 max_tokens: int = 128) -> list[int]:
        seq = self.add(prompt_token_ids, sampling_params, max_tokens)
        guard = (len(prompt_token_ids) + max_tokens) * 4 + 64
        while seq.status != SequenceStatus.FINISHED:
            finished = self.step()
            if not finished and self.is_idle():
                break
            guard -= 1
            assert guard > 0, "generate 疑似死循环"
        return seq.completion_token_ids

    def num_running(self) -> int:
        return len(self.scheduler.running)

    def reset_metrics(self):
        self.metrics.reset()

    def metrics_snapshot(self, gather: bool = False) -> dict:
        return self.metrics.snapshot(gather=gather)
    
