import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _rmsnorm_kernel(x_ptr, weight_ptr, out_ptr, n_cols: tl.constexpr,
                    eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask,
                other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    normalized = (x * tl.rsqrt(variance + eps)).to(tl.float16)
    weight = tl.load(weight_ptr + offsets, mask=mask,
                     other=0.0).to(tl.float16)
    tl.store(out_ptr + row * n_cols + offsets, normalized * weight, mask=mask)


@triton.jit
def _residual_rmsnorm_kernel(x_ptr, residual_ptr, weight_ptr, out_ptr,
                             residual_out_ptr, n_cols: tl.constexpr,
                             eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    base = row * n_cols + offsets
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + base, mask=mask,
                       other=0.0).to(tl.float32)
    summed = (x + residual).to(tl.float16)
    summed_fp32 = summed.to(tl.float32)
    variance = tl.sum(summed_fp32 * summed_fp32, axis=0) / n_cols
    normalized = (summed_fp32 * tl.rsqrt(variance + eps)).to(tl.float16)
    weight = tl.load(weight_ptr + offsets, mask=mask,
                     other=0.0).to(tl.float16)
    tl.store(out_ptr + base, normalized * weight, mask=mask)
    tl.store(residual_out_ptr + base, summed, mask=mask)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 use_triton: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.use_triton = use_triton
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        can_fuse = (self.use_triton and x.is_cuda and x.dtype == torch.float16
                    and x.is_contiguous() and x.shape[-1] == self.weight.numel())
        if can_fuse:
            x_2d = x.view(-1, x.shape[-1])
            out = torch.empty_like(x)
            block = triton.next_power_of_2(x.shape[-1])
            if residual is None:
                _rmsnorm_kernel[(x_2d.shape[0],)](
                    x, self.weight, out, x.shape[-1], self.eps,
                    BLOCK_SIZE=block)
                return out
            if residual.is_contiguous() and residual.shape == x.shape:
                residual_out = torch.empty_like(residual)
                _residual_rmsnorm_kernel[(x_2d.shape[0],)](
                    x, residual, self.weight, out, residual_out,
                    x.shape[-1], self.eps, BLOCK_SIZE=block)
                return out, residual_out

        if residual is None:
            return (self._norm(x.float()).to(x.dtype) * self.weight)

        y = (x.float() + residual.float()).to(x.dtype)
        return self._norm(y.float()).to(y.dtype) * self.weight, y
