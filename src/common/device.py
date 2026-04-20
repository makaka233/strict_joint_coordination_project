from __future__ import annotations
import torch

def resolve_device(device: str = "auto", gpu_id: int | None = None) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        if gpu_id is not None and 0 <= int(gpu_id) < torch.cuda.device_count():
            return torch.device(f"cuda:{int(gpu_id)}")
        return torch.device("cuda:0")
    return torch.device("cpu")

def describe_device(dev: torch.device) -> str:
    if dev.type == 'cuda':
        idx = dev.index if dev.index is not None else 0
        try:
            return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
        except Exception:
            return f"cuda:{idx}"
    return str(dev)
