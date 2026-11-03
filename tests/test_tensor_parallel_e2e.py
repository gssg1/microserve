import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from microserve.Config import Config
from microserve.distributed import destroy_distributed, get_device, is_rank_zero
from microserve.engine.llm import LLMEngine
from microserve.engine.sequence import SamplingParams


MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")
MAX_NEW_TOKENS = 12


def main():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = Config(
        model_dir=MODEL,
        enforce_eager=True,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        tensor_parallel_size=world_size,
    )
    engine = LLMEngine(config)
    prompts = [
        "The capital of France is",
        "2 + 2 =",
        "Once upon a time, there was a small",
    ]
    token_ids = [tokenizer(prompt)["input_ids"] for prompt in prompts]
    outputs = [
        engine.generate(ids, SamplingParams(temperature=0.0), MAX_NEW_TOKENS)
        for ids in token_ids
    ]

    passed = torch.ones(1, dtype=torch.int32, device=get_device())
    if is_rank_zero():
        reference = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16
        ).to(get_device()).eval()
        for prompt, ids, output in zip(prompts, token_ids, outputs):
            expected = reference.generate(
                torch.tensor([ids], device=get_device()),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )[0, len(ids):].tolist()
            same = output == expected
            passed &= int(same)
            print(f"prompt: {prompt!r}")
            print(f"microserve: {output}")
            print(f"transformers: {expected}")
            print(f"match: {same}")

    if dist.is_initialized():
        dist.broadcast(passed, src=0)
    assert passed.item() == 1
    if is_rank_zero():
        print(f"PASS: tensor_parallel_size={world_size}")


if __name__ == "__main__":
    try:
        main()
    finally:
        destroy_distributed()
