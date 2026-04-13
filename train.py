from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import build_local_change_aug, build_dataset
from models.vpas import build_model
from utils import (
    ConfigParser,
    barrier,
    choose_device,
    cleanup_distributed,
    is_main_process,
    reduce_mean,
    set_seed,
    setup_distributed,
    unwrap_model,
)
from utils.config_io import save_resolved_config
from val import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VPAS with config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config file.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda:0 or cpu.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training.")
    parser.add_argument("--dist_backend", type=str, default="nccl", help="Distributed backend, e.g. nccl or gloo.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by torch distributed launchers.")
    return parser.parse_args()


def _worker_init_fn(worker_id: int) -> None:
    del worker_id
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        import cv2

        cv2.setNumThreads(0)
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def _to_log_value(v: Any) -> Any:
    if isinstance(v, (str, bool, int, float)) or v is None:
        return v
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return float(v.detach().cpu().item())
        return str(v.shape)
    if isinstance(v, Path):
        return str(v)
    return str(v)


def _write_log(
    log_txt_path: Path,
    log_json_path: Path,
    log_history: List[Dict[str, Any]],
    text_line: str,
    record: Dict[str, Any],
) -> None:
    clean_record = {k: _to_log_value(v) for k, v in record.items()}
    clean_record["time"] = datetime.now().isoformat(timespec="seconds")

    log_history.append(clean_record)

    with log_txt_path.open("a", encoding="utf-8") as f:
        f.write(text_line.rstrip() + "\n")

    with log_json_path.open("w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)


def build_optimizer(cfg: ConfigParser, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg.optimizer
    name = str(opt_cfg.name).lower()

    lr = float(opt_cfg.lr)
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    if name == "adamw":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        nesterov = bool(opt_cfg.get("nesterov", False))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def get_trainable_parameter_info(model: torch.nn.Module) -> Tuple[List[str], int]:
    model_to_check = unwrap_model(model)
    trainable_named_params = [(name, param) for name, param in model_to_check.named_parameters() if param.requires_grad]
    trainable_names = [name for name, _ in trainable_named_params]
    trainable_count = int(sum(param.numel() for _, param in trainable_named_params))
    return trainable_names, trainable_count


def build_scheduler(
    cfg: ConfigParser,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
):
    sch_cfg = cfg.scheduler
    name = str(sch_cfg.name).lower()

    if name in {"none", "identity"}:
        return None

    if name == "cosine":
        epochs = int(cfg.train.epochs)
        warmup_steps_cfg = sch_cfg.get("warmup_steps", None)
        warmup_epochs = int(sch_cfg.get("warmup_epochs", 0))
        min_lr = float(sch_cfg.get("min_lr", 0.0))
        base_lr = float(cfg.optimizer.lr)
        min_lr_ratio = min_lr / max(base_lr, 1e-12)

        total_steps = max(int(epochs) * max(int(steps_per_epoch), 1), 1)
        if warmup_steps_cfg is not None:
            warmup_steps = max(int(warmup_steps_cfg), 0)
        else:
            warmup_steps = max(int(warmup_epochs) * max(int(steps_per_epoch), 1), 0)

        def lr_lambda(step_idx: int) -> float:
            if warmup_steps > 0 and step_idx < warmup_steps:
                return float(step_idx + 1) / float(warmup_steps)

            progress_total = max(total_steps - warmup_steps, 1)
            progress = min(max(step_idx - warmup_steps, 0), progress_total)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress / progress_total))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unsupported scheduler: {name}")


def create_dataloaders(
    cfg: ConfigParser,
    device: torch.device,
    num_workers_override: int | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DistributedSampler]]:
    local_change_aug = build_local_change_aug(cfg)
    train_dataset = build_dataset(cfg, split="train", local_change_aug=local_change_aug, return_mask=True)

    batch_size = int(cfg.train.batch_size)
    num_workers = int(num_workers_override) if num_workers_override is not None else int(cfg.data.get("num_workers", 4))
    val_interval = int(cfg.train.get("val_interval", 1))

    pin_memory = device.type == "cuda"
    persistent_workers = bool(cfg.data.get("persistent_workers", num_workers > 0)) and num_workers > 0
    prefetch_factor = int(cfg.data.get("prefetch_factor", 2)) if num_workers > 0 else None

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

    train_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": train_sampler is None,
        "sampler": train_sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "persistent_workers": persistent_workers,
        "worker_init_fn": _worker_init_fn if num_workers > 0 else None,
    }
    if prefetch_factor is not None:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )

    if val_interval == -1:
        return train_loader, None, train_sampler

    val_dataset = build_dataset(cfg, split="val", return_mask=True)
    val_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "persistent_workers": persistent_workers,
        "worker_init_fn": _worker_init_fn if num_workers > 0 else None,
    }
    if prefetch_factor is not None:
        val_loader_kwargs["prefetch_factor"] = prefetch_factor

    val_loader = DataLoader(
        val_dataset,
        **val_loader_kwargs,
    )

    return train_loader, val_loader, train_sampler


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    best_score: float,
    best_metric_name: str,
    best_f1_threshold: float | None = None,
) -> None:
    model_state = unwrap_model(model).state_dict()
    filtered_model_state = {k: v for k, v in model_state.items() if not k.startswith("backbone.")}
    ckpt = {
        "epoch": epoch,
        "model": filtered_model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "best_score": best_score,
        "best_metric_name": best_metric_name,
        "best_f1_threshold": best_f1_threshold,
        "excluded_state_dict_prefixes": ["backbone."],
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu",weights_only=True)

    model_to_load = unwrap_model(model)
    missing_keys, unexpected_keys = model_to_load.load_state_dict(ckpt["model"], strict=False)
    invalid_missing_keys = [k for k in missing_keys if not k.startswith("backbone.")]
    if invalid_missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint model state_dict mismatch. "
            f"Missing keys: {invalid_missing_keys}; unexpected keys: {unexpected_keys}"
        )
    optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    if ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_score = float(ckpt.get("best_score", ckpt.get("best_val_loss", float("-inf"))))
    return start_epoch, best_score


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    dataloader: DataLoader,
    device: torch.device,
    amp: bool,
    grad_clip: float,
    log_interval: int,
    epoch: int,
    writer: SummaryWriter | None = None,
    global_step_start: int = 0,
    show_progress: bool = True,
) -> Tuple[float, int]:
    model.train()

    use_amp = bool(amp) and device.type == "cuda"
    running_loss = 0.0
    total = 0

    pbar = tqdm(dataloader, desc=f"Train {epoch}", leave=True, dynamic_ncols=True, disable=not show_progress)
    for step, batch in enumerate(pbar, start=1):
        prompt = batch["prompt"].to(device, non_blocking=True)
        query = batch["query"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(prompt, query)
            loss = criterion(logits, mask)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        # Advance scheduler only when optimizer step is actually performed.
        # With AMP, scaler.step may skip optimizer.step on overflow.
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        if scheduler is not None and scale_after >= scale_before:
            scheduler.step()

        bsz = prompt.size(0)
        batch_loss = float(loss.detach().item())
        running_loss += batch_loss * bsz
        total += bsz

        epoch_loss = running_loss / max(total, 1)
        lr = float(optimizer.param_groups[0]["lr"])
        global_step = global_step_start + step
        if writer is not None:
            writer.add_scalar("train/step_loss", batch_loss, global_step)
            writer.add_scalar("train/lr", lr, global_step)
        pbar.set_postfix(batch_loss=f"{batch_loss:.4f}", epoch_loss=f"{epoch_loss:.4f}", lr=f"{lr:.2e}")

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"[train] epoch={epoch} step={step}/{len(dataloader)} "
                f"batch_loss={batch_loss:.6f} epoch_loss={epoch_loss:.6f} lr={lr:.8f}"
            )

    return running_loss / max(total, 1), global_step_start + len(dataloader)


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank = setup_distributed(args)

    try:
        cfg = ConfigParser.from_file(args.config)

        if args.output_dir is not None:
            cfg.set("experiment.output_dir", args.output_dir)
        if args.device is not None:
            cfg.set("experiment.device", args.device)
        if args.seed is not None:
            cfg.set("experiment.seed", int(args.seed))
        if args.batch_size is not None:
            cfg.set("train.batch_size", int(args.batch_size))

        seed = int(cfg.experiment.get("seed", 42))
        set_seed(seed + rank)

        device = choose_device(str(cfg.experiment.get("device", "cuda")), distributed=distributed, local_rank=local_rank)
        main_process = is_main_process(rank)

        disable_cudnn_cfg = bool(cfg.experiment.get("disable_cudnn", False))
        disable_cudnn_env = str(os.environ.get("DISABLE_CUDNN", "0")).lower() in {"1", "true", "yes", "on"}
        disable_cudnn = disable_cudnn_cfg or disable_cudnn_env
        if disable_cudnn:
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False

        output_dir = Path(str(cfg.experiment.get("output_dir", "./outputs/default")))
        if main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_resolved_config(cfg, output_dir / "resolved_config.yaml")
        barrier(device)

        log_txt_path = output_dir / "log.txt"
        log_json_path = output_dir / "log.json"
        tb_dir = output_dir / "tensorboard"
        log_history: List[Dict[str, Any]] = []

        writer = SummaryWriter(log_dir=str(tb_dir)) if main_process else None
        if main_process:
            log_txt_path.write_text("", encoding="utf-8")
            log_json_path.write_text("[]\n", encoding="utf-8")

        setup_msg_1 = f"[setup] device={device} seed={seed} rank={rank} world_size={world_size} distributed={distributed}"
        setup_msg_2 = f"[setup] output_dir={output_dir.resolve()}"
        setup_msg_3 = f"[setup] cudnn_enabled={torch.backends.cudnn.enabled}"
        if main_process:
            print(setup_msg_1)
            print(setup_msg_2)
            print(setup_msg_3)
            _write_log(
                log_txt_path,
                log_json_path,
                log_history,
                setup_msg_1,
                {"event": "setup", "device": str(device), "seed": seed, "rank": rank, "world_size": world_size, "distributed": distributed},
            )
            _write_log(
                log_txt_path,
                log_json_path,
                log_history,
                setup_msg_2,
                {"event": "setup", "output_dir": str(output_dir.resolve())},
            )
            _write_log(
                log_txt_path,
                log_json_path,
                log_history,
                setup_msg_3,
                {"event": "setup", "cudnn_enabled": bool(torch.backends.cudnn.enabled), "disable_cudnn": disable_cudnn},
            )

        if main_process:
            print("[setup] start build_model", flush=True)
        model, criterion = build_model(cfg)
        if main_process:
            print("[setup] finish build_model", flush=True)
        model = model.to(device)
        criterion = criterion.to(device)

        if distributed:
            ddp_kwargs: Dict[str, Any] = {}
            if device.type == "cuda":
                ddp_kwargs["device_ids"] = [local_rank]
                ddp_kwargs["output_device"] = local_rank
            model = DistributedDataParallel(model, **ddp_kwargs)

        trainable_param_names, trainable_param_count = get_trainable_parameter_info(model)
        if main_process:
            trainable_summary_msg = (
                f"[setup] trainable_parameter_tensors={len(trainable_param_names)} "
                f"trainable_parameter_count={trainable_param_count}"
            )
            print(trainable_summary_msg)
            _write_log(
                log_txt_path,
                log_json_path,
                log_history,
                trainable_summary_msg,
                {
                    "event": "setup",
                    "trainable_parameter_tensors": len(trainable_param_names),
                    "trainable_parameter_count": trainable_param_count,
                    "trainable_parameter_names": trainable_param_names,
                },
            )
            for param_name in trainable_param_names:
                print(f"[trainable_param] {param_name}")

        optimizer = build_optimizer(cfg, model)

        train_loader, val_loader, train_sampler = create_dataloaders(
            cfg,
            device,
            num_workers_override=args.num_workers,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

        amp = bool(cfg.train.get("amp", True))
        grad_clip = float(cfg.train.get("grad_clip", 0.0))
        log_interval = int(cfg.train.get("log_interval", 20))
        epochs = int(cfg.train.epochs)
        val_interval = int(cfg.train.get("val_interval", 1))

        best_metric_name = str(cfg.train.get("best_metric", "pixel_f1"))

        scaler = torch.amp.GradScaler(device.type, enabled=(amp and device.type == "cuda"))

        start_epoch = 1
        best_score = float("-inf")
        best_f1_threshold: float | None = None
        global_step = 0

        if args.resume:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            start_epoch, best_score = load_checkpoint(resume_path, model, optimizer, scheduler, scaler)
            global_step = max(start_epoch - 1, 0) * len(train_loader)
            resume_msg = f"[resume] from={resume_path} start_epoch={start_epoch} best_{best_metric_name}={best_score:.6f}"
            if main_process:
                print(resume_msg)
                _write_log(
                    log_txt_path,
                    log_json_path,
                    log_history,
                    resume_msg,
                    {
                        "event": "resume",
                        "resume_path": str(resume_path),
                        "start_epoch": start_epoch,
                        "best_metric_name": best_metric_name,
                        "best_score": best_score,
                    },
                )

        threshold = float(cfg.inference.get("threshold", None))

        if val_interval == -1 and main_process:
            skip_val_msg = "[setup] validation disabled because train.val_interval=-1"
            print(skip_val_msg)
            _write_log(
                log_txt_path,
                log_json_path,
                log_history,
                skip_val_msg,
                {"event": "setup", "validation_disabled": True, "val_interval": val_interval},
            )

        for epoch in range(start_epoch, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_loss, global_step = train_one_epoch(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                dataloader=train_loader,
                device=device,
                amp=amp,
                grad_clip=grad_clip,
                log_interval=log_interval if main_process else 0,
                epoch=epoch,
                writer=writer,
                global_step_start=global_step,
                show_progress=main_process,
            )
            train_loss = reduce_mean(train_loss, device, distributed)
            barrier(device)

            if main_process:
                log_msg = f"[epoch {epoch:03d}] train_loss={train_loss:.6f}"
                epoch_record: Dict[str, Any] = {
                    "event": "epoch",
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                if writer is not None:
                    writer.add_scalar("train/epoch_loss", float(train_loss), epoch)
                    writer.add_scalar("train/epoch_lr", float(optimizer.param_groups[0]["lr"]), epoch)

                do_val = val_loader is not None and val_interval > 0 and (epoch % val_interval == 0 or epoch == epochs)
                epoch_record["do_val"] = do_val

                if do_val:
                    val_metrics = validate(
                        model=model,
                        criterion=criterion,
                        dataloader=val_loader,
                        device=device,
                        amp=amp,
                        threshold=threshold,
                    )
                    epoch_record["val_metrics"] = val_metrics
                    if writer is not None:
                        for k, v in val_metrics.items():
                            if isinstance(v, (int, float)):
                                writer.add_scalar(f"val/{k}", float(v), epoch)

                    for k in val_metrics.keys():
                        v = val_metrics[k]
                        if isinstance(v, (int, float)):
                            log_msg += f" val_{k}={v:.6f}"
                    if "pixel_f1_threshold" in val_metrics and isinstance(val_metrics["pixel_f1_threshold"], (int, float)):
                        log_msg += f" val_pixel_f1_threshold={float(val_metrics['pixel_f1_threshold']):.6f}"

                    score = float(val_metrics.get(best_metric_name, float("nan")))
                    epoch_record["selection_score"] = score

                    if not math.isnan(score) and score > best_score:
                        best_score = score
                        thr_val = val_metrics.get("pixel_f1_threshold", None)
                        if isinstance(thr_val, (int, float)) and not math.isnan(float(thr_val)):
                            best_f1_threshold = float(thr_val)

                        epoch_record["is_best"] = True
                        epoch_record["best_score"] = best_score
                        epoch_record["best_f1_threshold"] = best_f1_threshold

                        save_checkpoint(
                            output_dir / "best.pth",
                            epoch,
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            best_score,
                            best_metric_name,
                            best_f1_threshold,
                        )
                    else:
                        epoch_record["is_best"] = False
                        epoch_record["best_score"] = best_score
                        epoch_record["best_f1_threshold"] = best_f1_threshold

                save_checkpoint(
                    output_dir / "last.pth",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    best_score,
                    best_metric_name,
                    best_f1_threshold,
                )

                print(log_msg)
                _write_log(log_txt_path, log_json_path, log_history, log_msg, epoch_record)

            barrier(device)

        if main_process:
            done_msg = f"[done] best_{best_metric_name}={best_score:.6f}"
            print(done_msg)
            if writer is not None:
                writer.close()
            _write_log(
                log_txt_path,
                log_json_path,
                log_history,
                done_msg,
                {
                    "event": "done",
                    "best_metric_name": best_metric_name,
                    "best_score": best_score,
                    "best_f1_threshold": best_f1_threshold,
                },
            )
        elif writer is not None:
            writer.close()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
