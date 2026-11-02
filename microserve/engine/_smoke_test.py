import itertools

from microserve.engine.sequence import Sequence, SequenceStatus
from microserve.engine.scheduler import Scheduler

BS = 4

def make_scheduler(num_blocks, max_batched, max_num_seqs=8, eos=None):
    return Scheduler(num_kvcache_blocks=num_blocks, block_size=BS,
                     max_num_seqs=max_num_seqs,
                     max_num_batched_tokens=max_batched,
                     eos_token_id=eos)

class FakeRunner:
    """模拟 model_runner.execute：返回递增 token（不命中 eos），校验执行区间。"""

    def __init__(self):
        self.tokens = itertools.count(20000)
        self.total_tokens_executed = 0
        self.first_start = {}

    def execute(self, batch):
        outputs = []
        for seq in batch:
            start, n = seq.num_computed_tokens, seq.num_scheduled_tokens
            assert n >= 1 and start + n <= len(seq.token_ids), "执行区间越界"
            if seq.gap == 1:
                assert start + n == len(seq.token_ids), "decode 区间必须覆盖最后 token"
            self.first_start.setdefault(seq.seq_id, start)
            self.total_tokens_executed += n
            outputs.append(None if not seq.needs_sample else next(self.tokens))
        return outputs

def drive(scheduler, runner, until=None, verbose=False):
    """驱动循环，跑满 until 轮（None=直到空闲）；返回 (轮数, 各轮 batch 快照)。"""
    rounds = []
    while not scheduler.is_idle() and (until is None or len(rounds) < until):
        batch = scheduler.schedule()
        if batch is None:
            rounds.append(None)
            break
        rounds.append([(s.seq_id, s.gap, s.num_scheduled_tokens, s.needs_sample)
                       for s in batch])
        if verbose:
            bm = scheduler.block_manager
            print(f"round {len(rounds)}: batch={[s.seq_id for s in batch]} "
                  f"free={bm.free_blocks_nums} cached={len(bm.hash_to_block)}")
        outputs = runner.execute(batch)
        scheduler.postprocess(batch, outputs)
    return len(rounds), rounds

def check_all_done(scheduler, seqs):
    assert scheduler.is_idle(), "调度器未到空闲态"
    for s in seqs:
        assert s.status == SequenceStatus.FINISHED, f"seq{s.seq_id} 未结束"
        assert not s.block_table, f"seq{s.seq_id} 块表未清空"
    bm = scheduler.block_manager
    assert bm.free_blocks_nums == len(bm.blocks), "存在未回收的物理块(泄漏)"
    assert not bm.used_blocks, "used_blocks 未清空"

def scenario_a_single_seq():
    print("== A: 单序列，预算充足一轮 prefill + (max_tokens-1) 轮 decode ==")
    sch = make_scheduler(num_blocks=16, max_batched=64)
    s = Sequence(list(range(10)), max_tokens=3)
    sch.add(s)
    runner = FakeRunner()
    n, rounds = drive(sch, runner)
    assert n == 1 + (3 - 1), f"期望 {1+(3-1)} 轮，实际 {n}"
    assert rounds[0][0][1:] == (10, 10, True), "首轮应为整段 prefill 并采样"
    for r in rounds[1:]:
        assert r == [(s.seq_id, 1, 1, True)], "decode 轮 sched 必须为 1"
    check_all_done(sch, [s])
    assert s.num_completion_tokens == 3 and s.num_tokens_with_spec == 13
    print(f"  OK（{n} 轮，共执行 {runner.total_tokens_executed} tokens）\n")

def scenario_b_chunked_prefill():
    print("== B: 分块 prefill（每轮预算 6 < prompt 10）==")
    sch = make_scheduler(num_blocks=16, max_batched=6)
    s = Sequence(list(range(10)), max_tokens=2)
    sch.add(s)
    runner = FakeRunner()
    n, rounds = drive(sch, runner)
    assert n == 2 + (2 - 1), f"期望 3 轮（prefill 6+4 + 1 decode），实际 {n}"
    assert rounds[0][0][2] == 6 and rounds[1][0][2] == 4
    check_all_done(sch, [s])
    assert s.num_completion_tokens == 2
    print(f"  OK（prefill 分块 6+4 后 1 轮 decode，共执行 {runner.total_tokens_executed} tokens）\n")

def scenario_c_prefix_cache():
    print("== C: prefix cache（后者复用前者的已算块，免于重算）==")
    prompt = list(range(20))
    sch = make_scheduler(num_blocks=16, max_batched=64)
    a = Sequence(prompt, max_tokens=2)
    sch.add(a)
    runner = FakeRunner()
    drive(sch, runner)
    assert len(sch.block_manager.hash_to_block) >= 4

    b = Sequence(prompt, max_tokens=2)
    sch.add(b)
    drive(sch, runner)

    assert runner.first_start[b.seq_id] == 16, \
        f"b 首次执行起点 {runner.first_start[b.seq_id]}，期望 16（cache 命中 4 块）"
    check_all_done(sch, [a, b])
    print(f"  OK（b 前 16 tokens 全部命中 cache，仅重算 {runner.total_tokens_executed - 22} tokens）\n")

def scenario_d_preemption():
    print("== D: 抢占（两个长 decode 争 8 块容量，到块边界时必然互相抢占）==")
    sch = make_scheduler(num_blocks=8, max_batched=8)
    seqs = [Sequence(list(range(12)), max_tokens=20),
            Sequence(list(range(12)), max_tokens=20)]
    for s in seqs:
        sch.add(s)
    calls = {"n": 0}
    orig = sch.preempt

    def counting_preempt(seq):
        calls["n"] += 1
        orig(seq)

    sch.preempt = counting_preempt
    runner = FakeRunner()
    n, _ = drive(sch, runner)
    assert calls["n"] > 0, "未发生抢占，测试场景无效"
    check_all_done(sch, seqs)
    print(f"  OK（抢占 {calls['n']} 次，{n} 轮完成，共执行 {runner.total_tokens_executed} tokens）\n")

def scenario_e_mixed_batch():
    print("== E: 混合批（同批既有 decode(gap=1) 又有 prefill(gap>1)）==")
    sch = make_scheduler(num_blocks=64, max_batched=24, max_num_seqs=16)
    s1 = Sequence(list(range(30)), max_tokens=10)
    sch.add(s1)
    runner = FakeRunner()

    drive(sch, runner, until=2)
    assert s1.gap == 1, "s1 应已进入 decode"
    s2 = Sequence(list(range(30)), max_tokens=2)
    sch.add(s2)
    n, rounds = drive(sch, runner)
    mixed = any(len(b) >= 2 and any(x[1] == 1 for x in b) and any(x[1] > 1 for x in b)
                for b in rounds if b)
    assert mixed, f"未观察到混合批。轮次: {rounds}"
    check_all_done(sch, [s1, s2])
    print(f"  OK（观察到 decode+prefill 同批；s1/s2 均完成）\n")

if __name__ == "__main__":
    scenario_a_single_seq()
    scenario_b_chunked_prefill()
    scenario_c_prefix_cache()
    scenario_d_preemption()
    scenario_e_mixed_batch()
    print("全部场景通过 ✔")
