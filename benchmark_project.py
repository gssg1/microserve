import argparse
import gc
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch
import triton

from microserve.Config import Config
from microserve.distributed import destroy_distributed, is_rank_zero
from microserve.engine.llm import LLMEngine
from microserve.engine.sequence import SamplingParams
from microserve.layers.layernorm import RMSNorm


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * pct / 100
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def summary(values):
    return {
        "mean": statistics.mean(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def prompts(seed, count, min_len, max_len, vocab_size):
    rng = random.Random(seed)
    return [
        [rng.randrange(100, min(vocab_size, 10000))
         for _ in range(rng.randint(min_len, max_len))]
        for _ in range(count)
    ]


def batch_kind(stats):
    if stats["prefill"] and stats["decode"]:
        return "mixed"
    return "prefill_only" if stats["prefill"] else "decode_only"


def run_burst(engine, seed, request_count=64, input_min=64, input_max=256,
              output_len=64):
    work = prompts(seed, request_count, input_min, input_max,
                   engine.runner.vocab_size)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    seqs = [engine.add(p, SamplingParams(temperature=0.0), output_len)
            for p in work]
    batches = {"prefill_only": 0, "decode_only": 0, "mixed": 0}
    started = time.perf_counter()
    while not engine.is_idle():
        engine.step()
        stats = engine.scheduler.last_batch_stats
        if stats:
            batches[batch_kind(stats)] += 1
    torch.cuda.synchronize()
    ended = time.perf_counter()
    ttft = [(s.token_timestamps[0] - s.arrived_at) * 1000 for s in seqs]
    itl = [(b - a) * 1000 for s in seqs
           for a, b in zip(s.token_timestamps, s.token_timestamps[1:])]
    lat = [(s.finished_at - s.arrived_at) * 1000 for s in seqs]
    output_tokens = sum(s.num_completion_tokens for s in seqs)
    return {
        "elapsed_s": ended - started,
        "output_tokens_per_s": output_tokens / (ended - started),
        "requests_per_s": request_count / (ended - started),
        "ttft_ms": summary(ttft),
        "itl_ms": summary(itl),
        "request_latency_ms": summary(lat),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "batch_counts": batches,
        "completion_token_ids": [s.completion_token_ids for s in seqs],
    }


def run_interference(engine, seed):
    background_prompts = prompts(seed, 8, 64, 64, engine.runner.vocab_size)
    background = [engine.add(p, SamplingParams(temperature=0.0), 96)
                  for p in background_prompts]
    while min(s.num_completion_tokens for s in background) < 16:
        engine.step()

    long_prompts = prompts(seed + 100000, 4, 1536, 1536,
                           engine.runner.vocab_size)
    long_requests = [engine.add(p, SamplingParams(temperature=0.0), 16)
                     for p in long_prompts]
    injected_at = min(s.arrived_at for s in long_requests)
    batches = {"prefill_only": 0, "decode_only": 0, "mixed": 0}
    torch.cuda.reset_peak_memory_stats()
    while not engine.is_idle():
        engine.step()
        stats = engine.scheduler.last_batch_stats
        if stats:
            batches[batch_kind(stats)] += 1
    torch.cuda.synchronize()

    background_itl = []
    for seq in background:
        for a, b in zip(seq.token_timestamps, seq.token_timestamps[1:]):
            if b >= injected_at:
                background_itl.append((b - a) * 1000)
    long_ttft = [(s.token_timestamps[0] - s.arrived_at) * 1000
                 for s in long_requests]
    finished_at = max(s.finished_at for s in background + long_requests)
    generated_after_injection = sum(
        sum(t >= injected_at for t in s.token_timestamps)
        for s in background + long_requests)
    return {
        "elapsed_after_injection_s": finished_at - injected_at,
        "output_tokens_per_s_after_injection": (
            generated_after_injection / (finished_at - injected_at)),
        "background_decode_itl_ms": summary(background_itl),
        "long_prompt_ttft_ms": summary(long_ttft),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "batch_counts": batches,
    }


def aggregate_trials(trials):
    return trials


def make_engine(args, *, policy="mixed", eager=False, fused=True,
                kv_dtype="fp16", quantization=None):
    return LLMEngine(Config(
        model_dir=args.model,
        max_num_seqs=16,
        max_num_batched_tokens=256,
        max_model_len=2048,
        num_kvcache_blocks=64,
        scheduler_policy=policy,
        enforce_eager=eager,
        fused_rmsnorm=fused,
        kv_cache_dtype=kv_dtype,
        quantization=quantization,
        eos_token_id=None,
        seed=0,
        tensor_parallel_size=args.tensor_parallel_size,
    ))


def destroy_engine(engine):
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def kv_cache_storage_bytes(engine):
    total = engine.runner.kv_cache.untyped_storage().nbytes()
    if engine.runner.kv_scale_cache is not None:
        total += engine.runner.kv_scale_cache.untyped_storage().nbytes()
    return total


def microbenchmark_rmsnorm():
    result = {}
    for rows in (1, 16, 256):
        x = torch.randn((rows, 1024), device="cuda", dtype=torch.float16)
        residual = torch.randn_like(x)
        pair = {}
        for fused in (False, True):
            layer = RMSNorm(1024, use_triton=fused).cuda().half()
            for _ in range(10):
                layer(x, residual)
            ms = triton.testing.do_bench(lambda: layer(x, residual),
                                         warmup=100, rep=500)
            pair["fused" if fused else "pytorch"] = ms
        pair["speedup"] = pair["pytorch"] / pair["fused"]
        result[str(rows)] = pair
    return result


def summarize_suite(raw):
    def med(section, metric, nested=None):
        values = []
        for trial in raw[section]:
            value = trial[metric]
            if nested:
                value = value[nested]
            values.append(value)
        return statistics.median(values)

    baseline = med("baseline_burst", "output_tokens_per_s")
    unfused = med("unfused_burst", "output_tokens_per_s")
    eager = med("eager_burst", "output_tokens_per_s")
    fp8_linear = med("fp8_linear_burst", "output_tokens_per_s")
    fp16_kv = med("fp16_kv_long_context", "output_tokens_per_s")
    fp8_kv = med("fp8_kv_long_context", "output_tokens_per_s")
    mixed_itl = med("mixed_interference", "background_decode_itl_ms", "p99")
    prefill_itl = med("prefill_first_interference",
                      "background_decode_itl_ms", "p99")
    mixed_ttft = med("mixed_interference", "long_prompt_ttft_ms", "p50")
    prefill_ttft = med("prefill_first_interference",
                       "long_prompt_ttft_ms", "p50")
    return {
        "burst_output_tokens_per_s_median": {
            "baseline_fused_graph": baseline,
            "unfused_graph": unfused,
            "eager_fused": eager,
            "fp8_linear_fused_graph": fp8_linear,
        },
        "relative_to_baseline": {
            "fused_rmsnorm_speedup": baseline / unfused,
            "cuda_graph_speedup": baseline / eager,
            "fp8_linear_speedup": fp8_linear / baseline,
        },
        "fp8_kv_long_context": {
            "fp16_output_tokens_per_s": fp16_kv,
            "fp8_output_tokens_per_s": fp8_kv,
            "fp8_speedup": fp8_kv / fp16_kv,
            "fp16_cache_bytes": raw["kv_cache_storage_bytes"]["fp16"],
            "fp8_cache_bytes": raw["kv_cache_storage_bytes"]["fp8_e4m3"],
            "cache_byte_reduction": 1 - (
                raw["kv_cache_storage_bytes"]["fp8_e4m3"] /
                raw["kv_cache_storage_bytes"]["fp16"]),
        },
        "chunked_prefill_interference": {
            "mixed_background_decode_itl_p99_ms": mixed_itl,
            "prefill_first_background_decode_itl_p99_ms": prefill_itl,
            "decode_itl_p99_reduction": 1 - mixed_itl / prefill_itl,
            "mixed_long_prompt_ttft_p50_ms": mixed_ttft,
            "prefill_first_long_prompt_ttft_p50_ms": prefill_ttft,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="project_benchmark.json")
    parser.add_argument(
        "--section",
        choices=("all", "baseline", "unfused", "eager", "prefill_first",
                 "fp8_linear", "fp8_kv"),
        default="all",
        help="Run one isolated group so each configuration gets a fresh process",
    )
    args = parser.parse_args()
    args.model = str(Path(args.model).expanduser())
    raw = {"environment": {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "repeats": args.repeats,
        "burst_workload": "64 requests, uniform 64-256 input, 64 output",
        "interference_workload": (
            "8 active decode requests; inject 4x1536-token prompts; "
            "256-token scheduler budget"),
    }}
    if args.section != "all":
        if args.section == "baseline":
            raw["rmsnorm_microbenchmark_ms"] = microbenchmark_rmsnorm()
            engine = make_engine(args)
            run_burst(engine, 9000, request_count=16, output_len=8)
            raw["baseline_burst"] = [run_burst(engine, 1000 + i)
                                     for i in range(args.repeats)]
            run_burst(engine, 9050, request_count=8, input_min=512,
                      input_max=1024, output_len=8)
            raw["fp16_kv_long_context"] = [
                run_burst(engine, 7000 + i, request_count=16,
                          input_min=512, input_max=1024, output_len=32)
                for i in range(args.repeats)]
            raw["kv_cache_storage_bytes"] = {
                "fp16": kv_cache_storage_bytes(engine)}
            run_interference(engine, 9100)
            raw["mixed_interference"] = [
                run_interference(engine, 2000 + i)
                for i in range(args.repeats)]
        elif args.section == "unfused":
            engine = make_engine(args, fused=False)
            run_burst(engine, 9200, request_count=16, output_len=8)
            raw["unfused_burst"] = [run_burst(engine, 1000 + i)
                                    for i in range(args.repeats)]
        elif args.section == "eager":
            engine = make_engine(args, eager=True)
            run_burst(engine, 9300, request_count=16, output_len=8)
            raw["eager_burst"] = [run_burst(engine, 1000 + i)
                                  for i in range(args.repeats)]
        elif args.section == "prefill_first":
            engine = make_engine(args, policy="prefill_first")
            run_interference(engine, 9400)
            raw["prefill_first_interference"] = [
                run_interference(engine, 2000 + i)
                for i in range(args.repeats)]
        elif args.section == "fp8_linear":
            engine = make_engine(args, quantization="fp8_dynamic")
            run_burst(engine, 9500, request_count=16, output_len=8)
            raw["fp8_linear_burst"] = [run_burst(engine, 1000 + i)
                                       for i in range(args.repeats)]
        else:
            engine = make_engine(args, kv_dtype="fp8_e4m3")
            run_burst(engine, 9600, request_count=8, input_min=512,
                      input_max=1024, output_len=8)
            raw["fp8_kv_long_context"] = [
                run_burst(engine, 7000 + i, request_count=16,
                          input_min=512, input_max=1024, output_len=32)
                for i in range(args.repeats)]
            raw["kv_cache_storage_bytes"] = {
                "fp8_e4m3": kv_cache_storage_bytes(engine)}
        if is_rank_zero():
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(raw, indent=2), encoding="utf-8")
            print("saved", output)
        return

    raw["rmsnorm_microbenchmark_ms"] = microbenchmark_rmsnorm()

    baseline = make_engine(args)
    run_burst(baseline, 9000, request_count=16, output_len=8)
    raw["baseline_burst"] = [run_burst(baseline, 1000 + i)
                             for i in range(args.repeats)]
    run_burst(baseline, 9050, request_count=8, input_min=512,
              input_max=1024, output_len=8)
    raw["fp16_kv_long_context"] = [
        run_burst(baseline, 7000 + i, request_count=16,
                  input_min=512, input_max=1024, output_len=32)
        for i in range(args.repeats)]
    raw["kv_cache_storage_bytes"] = {
        "fp16": kv_cache_storage_bytes(baseline),
    }
    run_interference(baseline, 9100)
    raw["mixed_interference"] = [run_interference(baseline, 2000 + i)
                                  for i in range(args.repeats)]
    destroy_engine(baseline)
    del baseline
    gc.collect()
    torch.cuda.empty_cache()

    unfused = make_engine(args, fused=False)
    run_burst(unfused, 9200, request_count=16, output_len=8)
    raw["unfused_burst"] = [run_burst(unfused, 1000 + i)
                            for i in range(args.repeats)]
    destroy_engine(unfused)
    del unfused
    gc.collect()
    torch.cuda.empty_cache()

    eager = make_engine(args, eager=True)
    run_burst(eager, 9300, request_count=16, output_len=8)
    raw["eager_burst"] = [run_burst(eager, 1000 + i)
                          for i in range(args.repeats)]
    destroy_engine(eager)
    del eager
    gc.collect()
    torch.cuda.empty_cache()

    prefill_first = make_engine(args, policy="prefill_first")
    run_interference(prefill_first, 9400)
    raw["prefill_first_interference"] = [
        run_interference(prefill_first, 2000 + i)
        for i in range(args.repeats)]
    destroy_engine(prefill_first)
    del prefill_first
    gc.collect()
    torch.cuda.empty_cache()

    fp8_linear = make_engine(args, quantization="fp8_dynamic")
    run_burst(fp8_linear, 9500, request_count=16, output_len=8)
    raw["fp8_linear_burst"] = [run_burst(fp8_linear, 1000 + i)
                               for i in range(args.repeats)]
    destroy_engine(fp8_linear)
    del fp8_linear
    gc.collect()
    torch.cuda.empty_cache()

    fp8_kv = make_engine(args, kv_dtype="fp8_e4m3")
    run_burst(fp8_kv, 9600, request_count=8, input_min=512,
              input_max=1024, output_len=8)
    raw["fp8_kv_long_context"] = [
        run_burst(fp8_kv, 7000 + i, request_count=16,
                  input_min=512, input_max=1024, output_len=32)
        for i in range(args.repeats)]
    raw["kv_cache_storage_bytes"]["fp8_e4m3"] = kv_cache_storage_bytes(fp8_kv)
    destroy_engine(fp8_kv)
    del fp8_kv
    gc.collect()
    torch.cuda.empty_cache()

    raw["summary"] = summarize_suite(raw)
    if is_rank_zero():
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        print(json.dumps(raw["summary"], indent=2))
        print("saved", output)


if __name__ == "__main__":
    try:
        main()
    finally:
        destroy_distributed()
