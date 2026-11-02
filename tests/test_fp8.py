import torch
import torch.nn.functional as F

from microserve.Config import Config
from microserve.engine.llm import LLMEngine
from microserve.engine.sequence import SamplingParams
from microserve.layers.fp8_linear import FP8DynamicLinear
from microserve.layers.linear import ReplicatedLinear


def main():
    torch.manual_seed(0)
    fp = ReplicatedLinear(50, 70).half().cuda()
    with torch.no_grad():
        fp.weight.normal_(0, 0.02)
    quant = FP8DynamicLinear(fp)
    assert quant.weight_fp8.dtype == torch.float8_e4m3fn
    for m in (1, 3, 17):
        x = torch.randn(m, 50, device="cuda", dtype=torch.float16)
        if m == 3:
            x[0].zero_()
        actual = quant(x)
        assert torch.isfinite(actual).all()
        assert (fp(x) - actual).abs().max().item() < 0.1
        sx = x.float().abs().amax(dim=1).clamp_min(1e-8) / 448
        xq = (x.float() / sx[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        reference = (F.linear(xq.float(), quant.weight_fp8.float()) *
                     sx[:, None] * quant.weight_scale[None, :]).half()
        assert (reference - actual).abs().max().item() < 0.002

    engine = LLMEngine(Config(quantization="fp8_dynamic", enforce_eager=False,
                              max_num_seqs=2, max_num_batched_tokens=64,
                              max_model_len=512))
    seq = engine.add([100, 101, 102, 103], SamplingParams(temperature=0.0),
                     max_tokens=3)
    while not engine.is_idle():
        engine.step()
    assert engine.runner.num_quantized_linears == 112
    assert len(seq.token_timestamps) == 3
    print("PASS: FP8 E4M3 GEMM, CUDA graph and timestamps")


if __name__ == "__main__":
    main()
