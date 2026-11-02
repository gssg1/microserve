from collections import deque
from time import perf_counter

from microserve.engine.block_manager import BlockManager
from microserve.engine.sequence import Sequence, SequenceStatus

class Scheduler:
    def __init__(self,
                 num_kvcache_blocks: int,
                 block_size: int,
                 max_num_seqs: int = 64,
                 max_num_batched_tokens: int = 4096,
                 policy: str = "mixed",
                 eos_token_id: int | None = None):
        self.block_manager = BlockManager(num_kvcache_blocks, block_size)
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos_token_id = eos_token_id
        self.policy = policy
        assert policy in ("mixed", "prefill_first")
        self.last_batch_stats: dict | None = None

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        self.model_runner = None

    def add(self, seq: Sequence):
        seq.arrived_at = seq.waiting_entered_at = perf_counter()
        self.waiting.append(seq)

    def is_idle(self) -> bool:
        return not self.waiting and not self.running

    def schedule(self) -> list[Sequence] | None:
        if not self.waiting and not self.running:
            self.last_batch_stats = None
            return None

        scheduled: list[Sequence] = []
        scheduled_ids: set[int] = set()
        budget = self.max_num_batched_tokens
        preempted: set[int] = set()

        if self.policy == "prefill_first":
            candidates = [s for s in self.running if s.gap > 1]
        else:
            candidates = self._running_snapshot(decode_first=True)

        for seq in candidates:
            if seq.gap <= 0 or budget <= 0:
                break
            if seq.seq_id in preempted:
                continue
            if self._advance(seq, budget, preempted, scheduled_ids):
                budget -= seq.num_scheduled_tokens
                scheduled.append(seq)
                scheduled_ids.add(seq.seq_id)
            else:
                budget = 0
                break

        while (self.waiting and budget > 0
               and len(self.running) < self.max_num_seqs):
            seq = self.waiting[0]
            assert not seq.block_table and seq.num_computed_tokens == 0
            ok = self.block_manager.allocate(seq, budget)
            if not ok or (seq.gap > 1 and seq.num_scheduled_tokens == 0):
                break
            self.waiting.popleft()
            self.running.append(seq)
            seq.status = SequenceStatus.RUNNING
            if seq.first_scheduled_at is None:
                seq.first_scheduled_at = perf_counter()
            if seq.gap == 1:
                seq.num_scheduled_tokens = 1
            seq.needs_sample = (
                seq.num_computed_tokens + seq.num_scheduled_tokens == seq.num_tokens_with_spec)
            budget -= seq.num_scheduled_tokens
            scheduled.append(seq)

        if self.policy == "prefill_first" and not scheduled:
            # Homogeneous, prefill-priority ablation inspired by nano-vLLM's
            # separate prefill/decode batches. It uses this engine's same KV
            # manager and runner, so the scheduler policy is the only change.
            budget = self.max_num_batched_tokens
            for seq in [s for s in self.running if s.gap == 1]:
                if budget <= 0:
                    break
                if self._advance(seq, budget, preempted, scheduled_ids):
                    budget -= seq.num_scheduled_tokens
                    scheduled.append(seq)
                    scheduled_ids.add(seq.seq_id)
                else:
                    break
        return self._record_batch(scheduled)

    def _record_batch(self, scheduled: list[Sequence]) -> list[Sequence] | None:
        if not scheduled:
            self.last_batch_stats = None
            return None
        prefill = [s for s in scheduled if s.gap > 1]
        decode = [s for s in scheduled if s.gap == 1]
        self.last_batch_stats = {
            "prefill": bool(prefill), "decode": bool(decode),
            "prefill_tokens": sum(s.num_scheduled_tokens for s in prefill),
            "decode_tokens": sum(s.num_scheduled_tokens for s in decode),
        }
        return scheduled

    def _advance(self, seq: Sequence, budget: int, preempted: set[int],
                 protected: set[int]) -> bool:

        is_decode = (seq.gap == 1)

        ok = self.block_manager.allocate(seq, budget)
        if not ok and self._preempt_victim(seq, preempted, protected):
            ok = self.block_manager.allocate(seq, budget)
        if ok and not is_decode and seq.num_scheduled_tokens == 0:
            return False
        if not ok or (not is_decode and seq.num_scheduled_tokens == 0):
            return False

        if is_decode:
            seq.num_scheduled_tokens = 1
        seq.needs_sample = (
            seq.num_computed_tokens + seq.num_scheduled_tokens == seq.num_tokens_with_spec)
        return True

    def _running_snapshot(self, decode_first: bool = True) -> list[Sequence]:
        """running 的稳定快照；decode(gap==1) 排前以保证其低延迟优先。"""
        seqs = list(self.running)
        if decode_first:
            seqs.sort(key=lambda s: 0 if s.gap == 1 else 1)
        return seqs

    def _preempt_victim(self, seq: Sequence, preempted: set[int],
                        protected: set[int]) -> bool:

        for victim in reversed(self.running):
            if (victim is seq or victim.seq_id in preempted
                    or victim.seq_id in protected):
                continue
            self.preempt(victim)
            preempted.add(victim.seq_id)
            return True
        return False

    def preempt(self, seq: Sequence):
        self.block_manager.deallocate(seq)
        seq.reset_for_preempt()
        if seq in self.running:
            self.running.remove(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], output_token_ids: list[int | None]):
        for seq, token_id in zip(seqs, output_token_ids):
            self.block_manager.hash_blocks(seq)

            seq.num_computed_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            if seq.num_computed_tokens < seq.num_tokens_with_spec:
                seq.needs_sample = False
                continue

            if token_id is None:
                continue
            seq.needs_sample = False
            sampled_at = perf_counter()
            if self.eos_token_id is not None and token_id == self.eos_token_id:
                self._finish(seq)
            else:
                seq.append_token(token_id)
                seq.token_timestamps.append(sampled_at)
                if seq.num_completion_tokens >= seq.max_tokens:
                    self._finish(seq)

    def _finish(self, seq: Sequence):
        seq.status = SequenceStatus.FINISHED
        seq.finished_at = perf_counter()
        self.block_manager.deallocate(seq)
        self.running.remove(seq)
