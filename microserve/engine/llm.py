from microserve.Config import Config
from microserve.engine.model_runner import ModelRunner
from microserve.engine.scheduler import Scheduler
from microserve.engine.sequence import SamplingParams, Sequence, SequenceStatus

class LLMEngine:
    def __init__(self, config: Config):
        self.config = config
        self.runner = ModelRunner(config)
        self.scheduler = Scheduler(
            num_kvcache_blocks=self.runner.num_blocks,
            block_size=config.block_size,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            policy=config.scheduler_policy,
            eos_token_id=config.eos_token_id)

    def add(self, token_ids: list[int],
            sampling_params: SamplingParams | None = None,
            max_tokens: int = 128) -> Sequence:
        seq = Sequence(token_ids, sampling_params=sampling_params,
                       max_tokens=max_tokens)
        self.scheduler.add(seq)
        return seq

    def step(self) -> list[Sequence]:
        """推进一轮调度+执行；返回本批完成（FINISHED）的 seq。"""
        batch = self.scheduler.schedule()
        if batch is None:
            return []
        outputs = self.runner.execute(batch)
        self.scheduler.postprocess(batch, outputs)
        return [s for s in batch if s.status == SequenceStatus.FINISHED]

    def is_idle(self) -> bool:
        return self.scheduler.is_idle()

    def generate(self, prompt_token_ids: list[int],
                 sampling_params: SamplingParams | None = None,
                 max_tokens: int = 128) -> list[int]:
        """同步跑完单个请求，返回生成的 token 序列（不含 prompt）。"""
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
    
