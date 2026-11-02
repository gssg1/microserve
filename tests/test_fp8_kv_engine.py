"""End-to-end FP8 KV mixed-batch and CUDA-graph regression."""

import os

from microserve.Config import Config
from microserve.engine.llm import LLMEngine
from microserve.engine.sequence import SamplingParams


MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")


def run(enforce_eager: bool):
    engine = LLMEngine(Config(
        model_dir=MODEL, kv_cache_dtype="fp8_e4m3",
        enforce_eager=enforce_eager, max_num_seqs=4,
        max_num_batched_tokens=32, num_kvcache_blocks=16,
        max_model_len=512))
    greedy = SamplingParams(temperature=0.0)
    first = engine.add(list(range(100, 108)), greedy, max_tokens=6)
    engine.step()
    second = engine.add(list(range(200, 297)), greedy, max_tokens=3)
    saw_mixed = saw_cached_chunk = False
    while not engine.is_idle():
        engine.step()
        stats = engine.scheduler.last_batch_stats
        if stats and stats["decode"] and stats["prefill"]:
            saw_mixed = True
            if 0 < second.num_computed_tokens < second.num_prompt_tokens:
                saw_cached_chunk = True
    assert saw_mixed and saw_cached_chunk
    return first.completion_token_ids, second.completion_token_ids


def main():
    eager = run(True)
    graph = run(False)
    assert eager == graph
    print("PASS: FP8 KV mixed execution and decode CUDA graph")


if __name__ == "__main__":
    main()
