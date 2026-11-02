"""Reproducible FP16 / FP8 workload and latency records."""

import argparse
import csv
import gc
import json
import random
import statistics
import time
from pathlib import Path

import torch

from microserve.Config import Config
from microserve.engine.llm import LLMEngine
from microserve.engine.sequence import SamplingParams


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summary(values):
    return {"mean": statistics.mean(values) if values else None,
            "p50": percentile(values, 50), "p95": percentile(values, 95),
            "p99": percentile(values, 99)}


def unique_model_storage_bytes(model):
    storages = {}
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor is not None:
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def run(args, quantization, prompts):
    torch.cuda.empty_cache()
    config = Config(model_dir=args.model, quantization=quantization,
                    kv_cache_dtype=args.kv_cache_dtype,
                    scheduler_policy=args.scheduler_policy,
                    num_kvcache_blocks=args.num_kvcache_blocks,
                    max_num_seqs=args.max_num_seqs,
                    max_num_batched_tokens=args.max_num_batched_tokens,
                    max_model_len=args.max_model_len,
                    enforce_eager=args.eager,
                    fused_rmsnorm=not args.no_fused_rmsnorm,
                    eos_token_id=None, seed=args.seed)
    engine = LLMEngine(config)
    assert all(max(p) < engine.runner.vocab_size for p in prompts)

    # Warm the same prefill/decode batch shapes before timing. Triton kernels
    # compile for shape M, so a single-request warmup would bias results.
    warm = [engine.add(ids, SamplingParams(temperature=0.0),
                       max_tokens=args.output_len) for ids in prompts]
    while not engine.is_idle():
        engine.step()
    assert all(s.num_completion_tokens == args.output_len for s in warm)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    seqs = [engine.add(ids, SamplingParams(temperature=0.0),
                       max_tokens=args.output_len) for ids in prompts]
    peak_waiting_requests = len(engine.scheduler.waiting)
    peak_running_seqs = len(engine.scheduler.running)
    batch_counts = {"prefill_only": 0, "decode_only": 0, "mixed": 0}
    scheduled_prefill_tokens = scheduled_decode_tokens = 0
    torch.cuda.nvtx.range_push("microserve_measured_" + (quantization or "fp16"))
    started = time.perf_counter()
    while not engine.is_idle():
        engine.step()
        stats = engine.scheduler.last_batch_stats
        if stats is not None:
            kind = ("mixed" if stats["prefill"] and stats["decode"] else
                    "prefill_only" if stats["prefill"] else "decode_only")
            batch_counts[kind] += 1
            scheduled_prefill_tokens += stats["prefill_tokens"]
            scheduled_decode_tokens += stats["decode_tokens"]
        peak_waiting_requests = max(peak_waiting_requests, len(engine.scheduler.waiting))
        peak_running_seqs = max(peak_running_seqs, len(engine.scheduler.running))
    torch.cuda.synchronize()
    ended = time.perf_counter()
    torch.cuda.nvtx.range_pop()

    rows = []
    for seq in seqs:
        stamps = seq.token_timestamps
        rows.append({
            "seq_id": seq.seq_id,
            "arrived_at": seq.arrived_at,
            "waiting_entered_at": seq.waiting_entered_at,
            "first_scheduled_at": seq.first_scheduled_at,
            "finished_at": seq.finished_at,
            "token_timestamps": stamps,
            "prompt_tokens": seq.num_prompt_tokens,
            "output_tokens": seq.num_completion_tokens,
            "completion_token_ids": seq.completion_token_ids,
            "preemption_count": seq.preemption_count,
            "ttft_ms": (stamps[0] - seq.arrived_at) * 1000 if stamps else None,
            "queue_ms": (seq.first_scheduled_at - seq.arrived_at) * 1000,
            "latency_ms": (seq.finished_at - seq.arrived_at) * 1000,
            "token_intervals_ms": [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])],
        })
    ttfts = [r["ttft_ms"] for r in rows if r["ttft_ms"] is not None]
    intervals = [x for r in rows for x in r["token_intervals_ms"]]
    elapsed = ended - started
    output_tokens = sum(r["output_tokens"] for r in rows)
    result = {
        "config": {"model": args.model, "quantization": quantization or "fp16",
                   "kv_cache_dtype": args.kv_cache_dtype,
                   "gpu": torch.cuda.get_device_name(), "seed": args.seed,
                   "requests": len(prompts), "input_len": args.input_len,
                   "input_len_min": args.input_len_min,
                   "output_len": args.output_len, "max_num_seqs": args.max_num_seqs,
                   "scheduler_policy": args.scheduler_policy,
                   "max_num_batched_tokens": args.max_num_batched_tokens,
                   "num_kvcache_blocks": engine.runner.num_blocks,
                   "eager": args.eager},
        "features": {"fused_rmsnorm": not args.no_fused_rmsnorm},
        "metrics": {"elapsed_s": elapsed, "requests_per_s": len(rows) / elapsed,
                    "output_tokens_per_s": output_tokens / elapsed,
                    "peak_waiting_requests": peak_waiting_requests,
                    "peak_running_seqs": peak_running_seqs,
                    "batch_counts": batch_counts,
                    "scheduled_prefill_tokens": scheduled_prefill_tokens,
                    "scheduled_decode_tokens": scheduled_decode_tokens,
                    "ttft_ms": summary(ttfts),
                    "queue_ms": summary([r["queue_ms"] for r in rows]),
                    "token_interval_ms": summary(intervals),
                    "latency_ms": summary([r["latency_ms"] for r in rows]),
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "model_unique_storage_bytes": unique_model_storage_bytes(
                        engine.runner.model),
                    "kv_cache_storage_bytes": (
                        engine.runner.kv_cache.untyped_storage().nbytes()
                        + (engine.runner.kv_scale_cache.untyped_storage().nbytes()
                           if engine.runner.kv_scale_cache is not None else 0)),
                    "quantized_linear_count": engine.runner.num_quantized_linears},
        "requests": rows,
    }
    del engine, warm, seqs
    gc.collect()
    torch.cuda.empty_cache()
    result["metrics"]["allocated_after_cleanup_bytes"] = torch.cuda.memory_allocated()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--input-len-min", type=int, default=None,
                        help="Uniform synthetic prompt lengths from this value to --input-len")
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--num-kvcache-blocks", type=int, default=64,
                        help="Fixed KV block count for comparable memory usage")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--no-fused-rmsnorm", action="store_true",
                        help="Use the unfused PyTorch RMSNorm reference path")
    parser.add_argument("--scheduler-policy", choices=["mixed", "prefill_first"],
                        default="mixed")
    parser.add_argument("--kv-cache-dtype", choices=["fp16", "fp8_e4m3"],
                        default="fp16")
    parser.add_argument("--reverse-order", action="store_true",
                        help="Run selected modes in reverse order to check order bias")
    parser.add_argument("--mode", choices=["both", "fp16", "fp8_dynamic"],
                        default="both")
    args = parser.parse_args()
    args.model = str(Path(args.model).expanduser())
    if args.input_len_min is None:
        args.input_len_min = args.input_len
    assert 0 < args.input_len_min <= args.input_len
    rng = random.Random(args.seed)
    prompts = [[rng.randrange(100, 10000) for _ in range(
                   rng.randint(args.input_len_min, args.input_len)
                   if args.input_len_min != args.input_len else args.input_len)]
               for _ in range(args.requests)]
    if args.mode == "both":
        modes = [None, "fp8_dynamic"]
    else:
        modes = [None if args.mode == "fp16" else args.mode]
    if args.reverse_order:
        modes.reverse()
    results = {mode or "fp16": run(args, mode, prompts) for mode in modes}
    if "fp16" in results and len(results) > 1:
        fp = results["fp16"]["requests"]
        results["comparison"] = {}
        for mode in results:
            if mode in ("fp16", "comparison"):
                continue
            q = results[mode]["requests"]
            results["comparison"][mode] = {
                "same_completion_fraction": sum(
                    a["completion_token_ids"] == b["completion_token_ids"]
                    for a, b in zip(fp, q)) / len(fp),
                "output_tokens_per_s_ratio_over_fp16": (
                    results[mode]["metrics"]["output_tokens_per_s"] /
                    results["fp16"]["metrics"]["output_tokens_per_s"]),
            }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with path.with_suffix(".tokens.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "seq_id", "token_index", "token_id", "timestamp_s", "since_arrival_ms"])
        for mode, result in results.items():
            if mode == "comparison":
                continue
            for row in result["requests"]:
                for i, (token, stamp) in enumerate(zip(row["completion_token_ids"],
                                                       row["token_timestamps"])):
                    writer.writerow([mode, row["seq_id"], i, token, stamp,
                                     (stamp - row["arrived_at"]) * 1000])
    for mode, result in results.items():
        if mode != "comparison":
            print(mode, json.dumps(result["metrics"], indent=2))
    if "comparison" in results:
        print("comparison", results["comparison"])
    print("saved", path, path.with_suffix(".tokens.csv"))


if __name__ == "__main__":
    main()
