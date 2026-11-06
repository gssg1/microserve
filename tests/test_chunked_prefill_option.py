from microserve.engine.scheduler import Scheduler
from microserve.engine.model_runner import ModelRunner
from microserve.engine.sequence import Sequence


def make_scheduler(enabled: bool, budget: int = 8):
    return Scheduler(
        num_kvcache_blocks=32,
        block_size=4,
        max_num_seqs=8,
        max_num_batched_tokens=budget,
        enable_chunked_prefill=enabled,
    )


def test_disabled_runs_prefill_and_decode_in_separate_batches():
    scheduler = make_scheduler(False)
    first = Sequence(list(range(4)), max_tokens=2)
    second = Sequence(list(range(10, 14)), max_tokens=2)
    scheduler.add(first)
    scheduler.add(second)

    batch = scheduler.schedule()
    assert batch == [first, second]
    assert all(seq.num_scheduled_tokens == 4 for seq in batch)
    scheduler.postprocess(batch, [100, 101])

    third = Sequence(list(range(20, 24)), max_tokens=2)
    scheduler.add(third)
    batch = scheduler.schedule()
    assert batch == [third]
    assert batch[0].num_scheduled_tokens == 4
    scheduler.postprocess(batch, [102])

    batch = scheduler.schedule()
    assert batch == [first, second, third]
    assert all(seq.gap == 1 and seq.num_scheduled_tokens == 1 for seq in batch)


def test_enabled_splits_prefill_by_token_budget():
    scheduler = make_scheduler(True, budget=6)
    seq = Sequence(list(range(10)), max_tokens=2)
    scheduler.add(seq)

    batch = scheduler.schedule()
    assert batch == [seq]
    assert seq.num_scheduled_tokens == 6
    scheduler.postprocess(batch, [None])

    batch = scheduler.schedule()
    assert batch == [seq]
    assert seq.num_computed_tokens == 6
    assert seq.num_scheduled_tokens == 4


def test_disabled_accepts_single_token_prompt():
    scheduler = make_scheduler(False)
    seq = Sequence([7], max_tokens=2)
    scheduler.add(seq)

    batch = scheduler.schedule()
    assert batch == [seq]
    assert seq.num_scheduled_tokens == 1
    assert seq.needs_sample


def test_disabled_allows_complete_remainder_after_prefix_cache_hit():
    scheduler = make_scheduler(False)
    first = Sequence(list(range(8)), max_tokens=2)
    scheduler.add(first)

    batch = scheduler.schedule()
    scheduler.postprocess(batch, [100])

    second = Sequence(list(range(8)), max_tokens=2)
    scheduler.add(second)
    batch = scheduler.schedule()

    assert batch == [second]
    assert second.num_computed_tokens == 4
    assert second.num_scheduled_tokens == 4
    assert not ModelRunner._is_partial_prefill(second)


def test_partial_prefill_detection_distinguishes_incomplete_work():
    seq = Sequence(list(range(10)), max_tokens=2)
    seq.num_scheduled_tokens = 6

    assert ModelRunner._is_partial_prefill(seq)

    seq.num_computed_tokens = 4
    seq.num_scheduled_tokens = 6

    assert not ModelRunner._is_partial_prefill(seq)
