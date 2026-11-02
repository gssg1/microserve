import random
import time

import torch

from microserve.Config import Config
from microserve.engine.llm import LLMEngine
from microserve.engine.sequence import SamplingParams


NUM_REQUESTS = 1000
INPUT_LEN = 128
OUTPUT_LEN = 64


config = Config(
    model_dir="~/huggingface/Qwen3-0.6B",
    max_num_seqs=8,
    max_num_batched_tokens=4096,
    max_model_len=4096,
    eos_token_id=None,
    enforce_eager=False,
)

engine = LLMEngine(config)
params = SamplingParams(temperature=0.0)

rng = random.Random(0)
vocab_size = engine.runner.vocab_size

requests = []

for _ in range(NUM_REQUESTS):
    token_ids = [
        rng.randrange(100, vocab_size)
        for _ in range(INPUT_LEN)
    ]

    seq = engine.add(
        token_ids,
        sampling_params=params,
        max_tokens=OUTPUT_LEN,
    )
    requests.append(seq)

torch.cuda.synchronize()
start = time.perf_counter()

while not engine.is_idle():
    engine.step()

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

input_tokens = sum(seq.num_prompt_tokens for seq in requests)
output_tokens = sum(seq.num_completion_tokens for seq in requests)

print(f"requests: {NUM_REQUESTS}")
print(f"time: {elapsed:.3f} s")
print(f"requests/s: {NUM_REQUESTS / elapsed:.2f}")
print(f"input tokens/s: {input_tokens / elapsed:.2f}")
print(f"output tokens/s: {output_tokens / elapsed:.2f}")
print(f"total tokens/s: {(input_tokens + output_tokens) / elapsed:.2f}")