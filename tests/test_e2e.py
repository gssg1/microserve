import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from microserve.Config import Config
from microserve.engine.llm import LLMEngine
from microserve.engine.sequence import SamplingParams, SequenceStatus

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")
MAX_NEW = 12

def run_prompts(prompts, chunk_budget):
    tok = AutoTokenizer.from_pretrained(MODEL)
    config = Config(model_dir=MODEL, enforce_eager=True,
                    max_num_seqs=4, max_num_batched_tokens=chunk_budget)
    config.eos_token_id = tok.eos_token_id
    eng = LLMEngine(config)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).to("cuda").eval()

    all_ok = True
    for text in prompts:
        ids = tok(text)["input_ids"]
        seq = eng.add(ids, sampling_params=SamplingParams(temperature=0.0),
                      max_tokens=MAX_NEW)
        guard = (len(ids) + MAX_NEW) * 4 + 64
        while seq.status != SequenceStatus.FINISHED:
            eng.step()
            if eng.is_idle():
                break
            guard -= 1
            assert guard > 0, "dead loop?"
        mine = seq.completion_token_ids
        ref = hf.generate(
            torch.tensor([ids], device="cuda"), max_new_tokens=MAX_NEW,
            do_sample=False, pad_token_id=tok.eos_token_id,
            use_cache=True)[0, len(ids):].tolist()
        same = mine == ref
        all_ok &= same
        print(f"prompt({len(ids)}tok): {text[:40]!r}")
        print(f"  microserve: {mine}")
        print(f"  hf        : {ref}")
        print(f"  match     : {same}")
        if not same:
            for i, (a, b) in enumerate(zip(mine, ref)):
                if a != b:
                    print(f"  first diff at #{i}: {a} vs {b}")
                    break
    return all_ok

def main():
    short_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "Once upon a time, there was a small",
    ]
    ok1 = run_prompts(short_prompts, chunk_budget=512)

    long_text = ("Quantum computing is a rapidly evolving field. " * 12).strip()
    ok2 = run_prompts([long_text], chunk_budget=32)

    print("PASS ✔" if (ok1 and ok2) else "FAIL ✘")

if __name__ == "__main__":
    main()
