import torch

from microserve.layers.layernorm import RMSNorm


def main():
    for shape in ((1, 1024), (17, 1024), (511, 1024)):
        torch.manual_seed(0)
        x = torch.randn(shape, device="cuda", dtype=torch.float16)
        residual = torch.randn_like(x)
        reference = RMSNorm(shape[-1], use_triton=False).cuda().half()
        fused = RMSNorm(shape[-1], use_triton=True).cuda().half()
        fused.weight.data.copy_(reference.weight)

        expected, expected_residual = reference(x, residual)
        actual, actual_residual = fused(x, residual)
        torch.testing.assert_close(actual_residual, expected_residual,
                                   atol=0, rtol=0)
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)

        expected = reference(x)
        actual = fused(x)
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    print("PASS: fused RMSNorm matches the PyTorch reference")


if __name__ == "__main__":
    main()
