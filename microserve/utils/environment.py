import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import torch


def package_version(name: str):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def command_output(command: list[str]):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=10)
        output = result.stdout.strip()
        if result.stderr.strip():
            output = "\n".join(part for part in (output, result.stderr.strip())
                               if part)
        return output or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment_info() -> dict:
    nccl_version = torch.cuda.nccl.version()
    if isinstance(nccl_version, tuple):
        nccl_version = list(nccl_version)
    gpus = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpus.append({
            "index": index,
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor],
        })
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": nccl_version,
        "triton": package_version("triton"),
        "flash_attn": package_version("flash-attn"),
        "gpus": gpus,
        "topology": command_output(["nvidia-smi", "topo", "-m"]),
        "driver": command_output([
            "nvidia-smi", "--query-gpu=driver_version",
            "--format=csv,noheader"]),
    }


def gpu_state():
    return command_output([
        "nvidia-smi",
        "--query-gpu=index,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ])


def server_report() -> dict:
    report = environment_info()
    report.update({
        "nvidia_smi": command_output(["nvidia-smi"]),
        "gpu_list": command_output(["nvidia-smi", "-L"]),
        "p2p_read": command_output(["nvidia-smi", "topo", "-p2p", "r"]),
        "p2p_write": command_output(["nvidia-smi", "topo", "-p2p", "w"]),
        "p2p_nvlink": command_output(["nvidia-smi", "topo", "-p2p", "n"]),
        "mig_mode": command_output([
            "nvidia-smi", "--query-gpu=index,mig.mode.current",
            "--format=csv,noheader"]),
        "compute_processes": command_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader"]),
        "lscpu": command_output(["lscpu"]),
        "numa": command_output(["numactl", "--hardware"]),
        "nsys": command_output(["nsys", "--version"]),
        "ncu": command_output(["ncu", "--version"]),
        "gpu_state": gpu_state(),
    })
    return report
