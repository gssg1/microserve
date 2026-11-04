import argparse
import os
import random
import time

import torch

from microserve.Config import Config
from microserve.distributed import destroy_distributed, is_rank_zero
from microserve.engine.llm import LLMEngine
from microserve.engine.sequence import SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--chunked-prefill", action="store_true")
    args = parser.parse_args()

    config = Config(
        model_dir="~/huggingface/Qwen3-0.6B",
        max_num_seqs=8,
        max_num_batched_tokens=4096,
        max_model_len=4096,
        eos_token_id=None,
        enforce_eager=args.eager,
        enable_chunked_prefill=args.chunked_prefill,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    engine = LLMEngine(config)
    params = SamplingParams(temperature=0.0)
    rng = random.Random(0)
    vocab_size = engine.runner.vocab_size
    requests = []
    for _ in range(args.requests):
        token_ids = [rng.randrange(100, vocab_size) for _ in range(args.input_len)]
        requests.append(engine.add(
            token_ids, sampling_params=params, max_tokens=args.output_len))

    torch.cuda.synchronize()
    start = time.perf_counter()
    while not engine.is_idle():
        engine.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if is_rank_zero():
        input_tokens = sum(seq.num_prompt_tokens for seq in requests)
        output_tokens = sum(seq.num_completion_tokens for seq in requests)
        print(f"requests: {args.requests}")
        print(f"time: {elapsed:.3f} s")
        print(f"requests/s: {args.requests / elapsed:.2f}")
        print(f"input tokens/s: {input_tokens / elapsed:.2f}")
        print(f"output tokens/s: {output_tokens / elapsed:.2f}")
        print(f"total tokens/s: {(input_tokens + output_tokens) / elapsed:.2f}")


if __name__ == "__main__":
    try:
        main()
    finally:
        destroy_distributed()
