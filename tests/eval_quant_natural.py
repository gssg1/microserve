import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from microserve.Config import Config
from microserve.engine.llm import LLMEngine
from microserve.engine.sequence import SamplingParams


MODEL = Path("~/huggingface/Qwen3-0.6B").expanduser()
PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time, there was a small",
]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    results = {}
    for mode in (None, "fp8_dynamic"):
        engine = LLMEngine(Config(model_dir=str(MODEL), quantization=mode,
                                  eos_token_id=tokenizer.eos_token_id,
                                  enforce_eager=False, max_num_seqs=3,
                                  max_num_batched_tokens=128,
                                  max_model_len=512))
        seqs = [engine.add(tokenizer(text)["input_ids"],
                           SamplingParams(temperature=0.0), max_tokens=16)
                for text in PROMPTS]
        while not engine.is_idle():
            engine.step()
        results[mode or "fp16"] = [
            {"prompt": text, "token_ids": seq.completion_token_ids,
             "text": tokenizer.decode(seq.completion_token_ids)}
            for text, seq in zip(PROMPTS, seqs)]
        del engine
        torch.cuda.empty_cache()
    rendered = json.dumps(results, indent=2, ensure_ascii=False)
    Path("quant_natural.json").write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
