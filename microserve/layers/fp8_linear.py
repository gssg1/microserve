import torch
from torch import nn
import triton
import triton.language as tl

from microserve.distributed import tensor_model_parallel_all_reduce
from microserve.layers.linear import LinearBase, RowParallelLinear
from microserve.utils.profiling import profile_range


FP8_MAX = 448.0


@triton.jit
def _quantize_fp8_rows(X, XQ, SX, M: tl.constexpr, K: tl.constexpr,
                       BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    x = tl.load(X + row * K + k, k < K, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0), 1e-8) / 448.0
    q = tl.clamp(x / scale, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(XQ + row * K + k, q, k < K)
    tl.store(SX + row, scale)


@triton.jit
def _fp8_mm(XQ, WQ, SX, SW, Y, M: tl.constexpr, N: tl.constexpr,
            K: tl.constexpr, BM: tl.constexpr = 16,
            BN: tl.constexpr = 64, BK: tl.constexpr = 32):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + ki
        xq = tl.load(XQ + mi[:, None] * K + k[None, :],
                     (mi[:, None] < M) & (k[None, :] < K), other=0.0)
        wq = tl.load(WQ + ni[None, :] * K + k[:, None],
                     (ni[None, :] < N) & (k[:, None] < K), other=0.0)
        acc = tl.dot(xq, wq, acc)
    sx = tl.load(SX + mi, mi < M, other=0)
    sw = tl.load(SW + ni, ni < N, other=0)
    y = acc * sx[:, None] * sw[None, :]
    tl.store(Y + mi[:, None] * N + ni[None, :], y,
             (mi[:, None] < M) & (ni[None, :] < N))


class FP8DynamicLinear(nn.Module):
    def __init__(self, source: LinearBase):
        super().__init__()
        weight = source.weight.detach().float()
        scales = weight.abs().amax(dim=1).clamp_min(1e-8) / FP8_MAX
        packed = (weight / scales[:, None]).clamp(-FP8_MAX, FP8_MAX).to(
            torch.float8_e4m3fn)
        self.register_buffer("weight_fp8", packed.contiguous())
        self.register_buffer("weight_scale", scales.contiguous())
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().clone())
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.reduce_results = isinstance(source, RowParallelLinear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float16 or not x.is_cuda:
            raise ValueError("fp8_dynamic requires CUDA FP16 input activations")
        x2 = x.contiguous().view(-1, self.in_features)
        m = x2.shape[0]
        xq = torch.empty(x2.shape, device=x.device, dtype=torch.float8_e4m3fn)
        sx = torch.empty(m, device=x.device, dtype=torch.float32)
        _quantize_fp8_rows[(m,)](
            x2, xq, sx, m, self.in_features,
            triton.next_power_of_2(self.in_features))
        y = torch.empty((m, self.out_features), device=x.device, dtype=x.dtype)
        _fp8_mm[(triton.cdiv(m, 16), triton.cdiv(self.out_features, 64))](
            xq, self.weight_fp8, sx, self.weight_scale, y,
            m, self.out_features, self.in_features)
        if self.reduce_results:
            with profile_range("row_parallel.all_reduce", detailed=True):
                y = tensor_model_parallel_all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y.view(*x.shape[:-1], self.out_features)


def quantize_fp8_linears(model: nn.Module) -> int:
    count = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LinearBase):
                setattr(parent, name, FP8DynamicLinear(child))
                count += 1
    return count
