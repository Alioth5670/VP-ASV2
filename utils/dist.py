from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def is_dist_enabled(args: argparse.Namespace) -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return bool(args.distributed or world_size > 1)


def setup_distributed(args: argparse.Namespace) -> Tuple[bool, int, int, int]:
    distributed = is_dist_enabled(args)
    if not distributed:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=args.dist_backend, init_method="env://")

    return True, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(device: torch.device | None = None) -> None:
    if dist.is_available() and dist.is_initialized():
        if device is not None and device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()


def is_main_process(rank: int) -> bool:
    return rank == 0


def choose_device(device_str: str | None, distributed: bool = False, local_rank: int = 0) -> torch.device:
    if device_str:
        if distributed and device_str.startswith("cuda") and torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device(device_str)
    if distributed and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def reduce_mean(value: float, device: torch.device, distributed: bool) -> float:
    if not distributed:
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())
