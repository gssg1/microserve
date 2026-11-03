import torch
import torch.nn.functional as F
from torch import nn

from microserve.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from microserve.utils.profiling import profile_range


def divide(value: int, divisor: int) -> int:
    assert value % divisor == 0, f"{value} is not divisible by {divisor}"
    return value // divisor


class LinearBase(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.bias_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def bias_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class ReplicatedLinear(LinearBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size = input_size
        self.input_size_per_partition = divide(input_size, self.tp_size)
        super().__init__(self.input_size_per_partition, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        start = self.tp_rank * self.input_size_per_partition
        shard = loaded_weight.narrow(1, start, self.input_size_per_partition)
        param.data.copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with profile_range("row_parallel.matmul", detailed=True):
            output = F.linear(x, self.weight, None)
        with profile_range("row_parallel.all_reduce", detailed=True):
            output = tensor_model_parallel_all_reduce(output)
        if self.bias is not None:
            output = output + self.bias
        return output


class MergedColumnParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.output_sizes = output_sizes
        self.output_sizes_per_partition = [
            divide(size, self.tp_size) for size in output_sizes
        ]
        super().__init__(input_size, sum(self.output_sizes_per_partition), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      loaded_shard_id: int):
        local_size = self.output_sizes_per_partition[loaded_shard_id]
        source_start = self.tp_rank * local_size
        target_start = sum(self.output_sizes_per_partition[:loaded_shard_id])
        source = loaded_weight.narrow(0, source_start, local_size)
        param.data.narrow(0, target_start, local_size).copy_(source)

    def bias_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                    loaded_shard_id: int):
        self.weight_loader(param, loaded_weight, loaded_shard_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class QKVParallelLinear(LinearBase):
    def __init__(self, hidden_size: int, head_size: int,
                 total_num_heads: int, total_num_kv_heads: int,
                 bias: bool = False):
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = divide(total_num_heads, self.tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, self.tp_size)
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      loaded_shard_id: str):
        assert loaded_shard_id in ("q", "k", "v")
        if loaded_shard_id == "q":
            local_size = self.num_heads * self.head_size
            target_start = 0
        elif loaded_shard_id == "k":
            local_size = self.num_kv_heads * self.head_size
            target_start = self.num_heads * self.head_size
        else:
            local_size = self.num_kv_heads * self.head_size
            target_start = (self.num_heads + self.num_kv_heads) * self.head_size
        source_start = self.tp_rank * local_size
        source = loaded_weight.narrow(0, source_start, local_size)
        param.data.narrow(0, target_start, local_size).copy_(source)

    def bias_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                    loaded_shard_id: str):
        self.weight_loader(param, loaded_weight, loaded_shard_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
