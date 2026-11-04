import torch
from dataclasses import dataclass, field

@dataclass
class Config:
    model_dir: str = "~/huggingface/Qwen3-0.6B"

    block_size: int = 256
    num_kvcache_blocks: int = 0
    gpu_memory_utilization: float = 0.90

    max_num_seqs: int = 32
    max_num_batched_tokens: int = 4096
    max_model_len: int = 4096
    eos_token_id: int | None = None

    dtype: torch.dtype = field(default=torch.float16)
    seed: int = 0
    enforce_eager: bool = False
    fused_rmsnorm: bool = True
    quantization: str | None = None
    kv_cache_dtype: str = "fp16"
    scheduler_policy: str = "mixed"
    enable_chunked_prefill: bool = False
    tensor_parallel_size: int = 1
    enable_metrics: bool = False
    enable_nvtx: bool = False
    profile_layers: bool = False
    profile_cuda_events: bool = False

    def __post_init__(self):
        import os
        self.model_dir = os.path.expanduser(self.model_dir)
        assert self.block_size > 0 and self.block_size % 256 == 0, \
            "flash-attn paged kernel 要求 block_size 是 256 的倍数"
        assert 0 < self.gpu_memory_utilization <= 1.0
        assert self.quantization in (None, "fp8_dynamic"), \
            "unsupported quantization"
        assert self.kv_cache_dtype in ("fp16", "fp8_e4m3"), \
            "unsupported KV cache dtype"
        if self.kv_cache_dtype == "fp8_e4m3":
            assert self.dtype == torch.float16, "FP8 KV cache currently requires FP16 model dtype"
        assert self.scheduler_policy in ("mixed", "prefill_first"), \
            "unsupported scheduler policy"
        assert self.tensor_parallel_size > 0
