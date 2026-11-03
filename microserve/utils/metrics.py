from collections import defaultdict

import torch
import torch.distributed as dist


class MetricsCollector:
    def __init__(self, enabled: bool = False, cuda_events: bool = False):
        self.enabled = enabled or cuda_events
        self.cuda_events = cuda_events
        self.reset()

    def reset(self):
        self.steps: list[dict] = []
        self.current: dict | None = None
        self.pending_events: list[tuple[dict, str, torch.cuda.Event,
                                        torch.cuda.Event]] = []
        self.next_step_id = 0

    def begin_step(self, values: dict):
        if not self.enabled:
            return
        self.current = {
            "step_id": self.next_step_id,
            "cpu_ms": {},
            "gpu_ms": {},
            **values,
        }
        self.next_step_id += 1

    def add_cpu(self, name: str, milliseconds: float):
        if self.current is None:
            return
        values = self.current["cpu_ms"]
        values[name] = values.get(name, 0.0) + milliseconds

    def add_value(self, name: str, value):
        if self.current is not None:
            self.current[name] = value

    def add_cuda_event(self, name: str, start: torch.cuda.Event,
                       end: torch.cuda.Event):
        if self.current is not None:
            self.pending_events.append((self.current, name, start, end))

    def end_step(self):
        if self.current is not None:
            self.steps.append(self.current)
            self.current = None

    def resolve_cuda_events(self):
        if not self.pending_events:
            return
        torch.cuda.synchronize()
        for step, name, start, end in self.pending_events:
            values = step["gpu_ms"]
            values[name] = values.get(name, 0.0) + start.elapsed_time(end)
        self.pending_events.clear()

    def snapshot(self, gather: bool = False) -> dict:
        self.resolve_cuda_events()
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        local = {"rank": rank, "steps": self.steps}
        ranks = [local]
        if gather and dist.is_available() and dist.is_initialized():
            ranks = [None] * dist.get_world_size()
            dist.all_gather_object(ranks, local)
        return self._aggregate(ranks)

    @staticmethod
    def _aggregate(ranks: list[dict]) -> dict:
        if not ranks:
            return {"ranks": [], "steps": [], "summary": {}}
        rank_steps = {
            rank["rank"]: {step["step_id"]: step for step in rank["steps"]}
            for rank in ranks
        }
        base_rank = min(rank_steps)
        merged_steps = []
        for step_id, base in rank_steps[base_rank].items():
            merged = {key: value for key, value in base.items()
                      if key not in ("cpu_ms", "gpu_ms")}
            merged["cpu_ms"] = {}
            merged["gpu_ms"] = {}
            cpu_names = set()
            gpu_names = set()
            for steps in rank_steps.values():
                step = steps.get(step_id, {})
                cpu_names.update(step.get("cpu_ms", {}))
                gpu_names.update(step.get("gpu_ms", {}))
            for name in cpu_names:
                merged["cpu_ms"][name] = max(
                    steps.get(step_id, {}).get("cpu_ms", {}).get(name, 0.0)
                    for steps in rank_steps.values())
            for name in gpu_names:
                merged["gpu_ms"][name] = max(
                    steps.get(step_id, {}).get("gpu_ms", {}).get(name, 0.0)
                    for steps in rank_steps.values())
            merged_steps.append(merged)

        cpu_totals = defaultdict(float)
        gpu_totals = defaultdict(float)
        for step in merged_steps:
            for name, value in step["cpu_ms"].items():
                cpu_totals[name] += value
            for name, value in step["gpu_ms"].items():
                gpu_totals[name] += value
        token_totals = {
            "prefill": sum(step.get("scheduled_prefill_tokens", 0)
                           for step in merged_steps),
            "first_prefill": sum(step.get("first_prefill_tokens", 0)
                                 for step in merged_steps),
            "chunk_prefill": sum(step.get("chunk_prefill_tokens", 0)
                                 for step in merged_steps),
            "decode": sum(step.get("scheduled_decode_tokens", 0)
                          for step in merged_steps),
        }
        throughput = {}
        stage_tokens = {
            "prefill.model": token_totals["first_prefill"],
            "chunk_prefill.model": token_totals["chunk_prefill"],
            "decode.model": token_totals["decode"],
        }
        for stage, tokens in stage_tokens.items():
            milliseconds = gpu_totals.get(stage, 0.0)
            if milliseconds > 0:
                throughput[stage] = tokens * 1000.0 / milliseconds
        summary = {
            "num_steps": len(merged_steps),
            "token_totals": token_totals,
            "cpu_ms": dict(sorted(cpu_totals.items())),
            "gpu_ms": dict(sorted(gpu_totals.items())),
            "stage_tokens_per_s": throughput,
        }
        return {"ranks": ranks, "steps": merged_steps, "summary": summary}
