from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count
from time import perf_counter

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1

class Sequence:
    _counter = count()

    def __init__(self,
                 token_ids: list[int],
                 sampling_params: SamplingParams | None = None,
                 max_tokens: int = 128,
                 seq_id: int | None = None):
        self.seq_id = next(Sequence._counter) if seq_id is None else seq_id
        self.status = SequenceStatus.WAITING
        self.waiting_entered_at: float | None = None
        self.arrived_at: float | None = None
        self.first_scheduled_at: float | None = None
        self.prefill_started_at: float | None = None
        self.prefill_finished_at: float | None = None
        self.finished_at: float | None = None
        self.token_timestamps: list[float] = []
        self.preemption_count = 0
        self.num_scheduler_steps = 0
        self.num_prefill_steps = 0
        self.num_decode_steps = 0
        self.token_ids = copy(token_ids)
        self.num_prompt_tokens = len(token_ids)

        self.num_computed_tokens = 0
        self.num_tokens_with_spec = len(token_ids)

        self.num_scheduled_tokens = 0
        self.needs_sample = False
        self.block_table: list[int] = []

        self.max_tokens = max_tokens
        self.sampling_params = sampling_params if sampling_params is not None else SamplingParams()

    @property
    def gap(self) -> int:
        return self.num_tokens_with_spec - self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens_with_spec - self.num_prompt_tokens

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.num_tokens_with_spec += 1

    def reset_for_preempt(self):
        self.status = SequenceStatus.WAITING
        self.waiting_entered_at = perf_counter()
        self.preemption_count += 1
        self.num_computed_tokens = 0
        self.num_scheduled_tokens = 0
        self.needs_sample = False
        self.block_table = []
