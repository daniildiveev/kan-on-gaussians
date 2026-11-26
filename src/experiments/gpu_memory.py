"""Resolve a CUDA device in its worker and query driver-reported memory in MiB."""

import ctypes
import os
import subprocess
from uuid import UUID

import torch


def selected_gpu_uuid(device: torch.device) -> str:
    """Use CUDA's visible-device order, including CUDA_VISIBLE_DEVICES remapping."""
    value = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    if value is not None:
        if isinstance(value, bytes) and len(value) == 16:
            return f"GPU-{UUID(bytes=value)}"
        return f"GPU-{UUID(str(value).removeprefix('GPU-'))}"

    # Older PyTorch builds omit uuid. The worker already initialized this CUDA context.
    driver = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
    handle = ctypes.c_int()
    raw = (ctypes.c_ubyte * 16)()
    if driver.cuDeviceGet(ctypes.byref(handle), device.index) or driver.cuDeviceGetUuid(ctypes.byref(raw), handle):
        raise RuntimeError(f"CUDA could not resolve the GPU UUID for {device}")
    return f"GPU-{UUID(bytes=bytes(raw))}"


def device_memory_mib(uuid: str) -> int:
    value = subprocess.check_output(
        ["nvidia-smi", f"--id={uuid}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True, timeout=5,
    ).strip()
    if not value.isdecimal():
        raise RuntimeError(f"nvidia-smi returned no usable memory reading for {uuid}: {value!r}")
    return int(value)


def gpu_memory_by_pid(pids: set[int]) -> dict[int, int]:
    """Read driver-reported MiB for our compute processes, including CUDA contexts."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        text=True, timeout=5,
    )
    usage = {}
    for line in output.splitlines():
        pid, _, memory = line.partition(",")
        if pid.strip().isdigit() and int(pid) in pids and memory.strip().isdecimal():
            pid = int(pid)
            usage[pid] = usage.get(pid, 0) + int(memory.strip())
    return usage
