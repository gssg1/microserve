import os

import torch
import torch.distributed as dist


_LOCAL_RANK = 0
_GLOBAL_RANK = 0
_WORLD_SIZE = 1
_DEVICE: torch.device | None = None
_TP_GROUP = None


def initialize_distributed() -> None:
    global _LOCAL_RANK, _GLOBAL_RANK, _WORLD_SIZE, _DEVICE, _TP_GROUP
    if dist.is_available() and dist.is_initialized():
        _GLOBAL_RANK = dist.get_rank()
        _WORLD_SIZE = dist.get_world_size()
        _LOCAL_RANK = int(os.environ.get("LOCAL_RANK", _GLOBAL_RANK))
        torch.cuda.set_device(_LOCAL_RANK)
        _DEVICE = torch.device("cuda", _LOCAL_RANK)
        _TP_GROUP = None
        return
    _LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    _GLOBAL_RANK = int(os.environ.get("RANK", 0))
    _WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(_LOCAL_RANK)
    _DEVICE = torch.device("cuda", _LOCAL_RANK)
    if _WORLD_SIZE > 1:
        dist.init_process_group(backend="nccl", device_id=_DEVICE)
        _GLOBAL_RANK = dist.get_rank()
        _WORLD_SIZE = dist.get_world_size()
    _TP_GROUP = None


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_local_rank() -> int:
    return _LOCAL_RANK


def get_global_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return _GLOBAL_RANK


def get_tensor_model_parallel_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=_TP_GROUP)
    return 0


def get_tensor_model_parallel_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group=_TP_GROUP)
    return 1


def get_device() -> torch.device:
    if _DEVICE is not None:
        return _DEVICE
    return torch.device("cuda", _LOCAL_RANK)


def is_rank_zero() -> bool:
    return get_global_rank() == 0


def tensor_model_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    if get_tensor_model_parallel_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=_TP_GROUP)
    return tensor


def tensor_model_parallel_min(value: int) -> int:
    if get_tensor_model_parallel_world_size() == 1:
        return value
    tensor = torch.tensor([value], dtype=torch.int64, device=get_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=_TP_GROUP)
    return int(tensor.item())


def gather_tensor(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return tensor
    parts = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(parts, tensor.contiguous(), group=_TP_GROUP)
    return torch.cat(parts, dim=dim)


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    if get_tensor_model_parallel_world_size() > 1:
        dist.broadcast(tensor, src=src, group=_TP_GROUP)
    return tensor
