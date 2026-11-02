import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoTokenizer
from microserve.Config import Config
from microserve.engine.llm import LLMEngine
from microserve.engine.sequence import SamplingParams, SequenceStatus

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")

def run(enforce_eager: bool, text: str, max_new: int):
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = Config(model_dir=MODEL, enforce_eager=enforce_eager,
                 max_num_seqs=4, max_num_batched_tokens=512)
    cfg.eos_token_id = tok.eos_token_id
    eng = LLMEngine(cfg)
    ids = tok(text)["input_ids"]
    seq = eng.add(ids, sampling_params=SamplingParams(temperature=0.0), max_tokens=max_new)
    guard = 10000
    while seq.status != SequenceStatus.FINISHED:
        eng.step()
        if eng.is_idle():
            break
        guard -= 1
        assert guard > 0
    return seq.completion_token_ids

def main():
    text = "The capital of France is"
    eager = run(True, text, 10)
    print("eager:", eager)
    graph = run(False, text, 10)
    print("graph:", graph)
    print("MATCH ✔" if eager == graph else "MISMATCH ✘")
    print("graph sizes:", len(run_graph_bs := [1, 2, 4]))

if __name__ == "__main__":
    main()
