from microserve.utils.metrics import MetricsCollector


def main():
    ranks = [
        {
            "rank": 0,
            "steps": [{
                "step_id": 0,
                "scheduled_prefill_tokens": 8,
                "scheduled_decode_tokens": 0,
                "first_prefill_tokens": 8,
                "chunk_prefill_tokens": 0,
                "cpu_ms": {"scheduler.schedule": 1.0},
                "gpu_ms": {"prefill.model": 4.0},
            }],
        },
        {
            "rank": 1,
            "steps": [{
                "step_id": 0,
                "scheduled_prefill_tokens": 8,
                "scheduled_decode_tokens": 0,
                "first_prefill_tokens": 8,
                "chunk_prefill_tokens": 0,
                "cpu_ms": {"scheduler.schedule": 1.5},
                "gpu_ms": {"prefill.model": 5.0},
            }],
        },
    ]
    result = MetricsCollector._aggregate(ranks)
    step = result["steps"][0]
    summary = result["summary"]
    assert step["cpu_ms"]["scheduler.schedule"] == 1.5
    assert step["gpu_ms"]["prefill.model"] == 5.0
    assert summary["token_totals"]["prefill"] == 8
    assert summary["stage_tokens_per_s"]["prefill.model"] == 1600.0
    print("PASS")


if __name__ == "__main__":
    main()
