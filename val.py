from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import build_dataset
from models.vpas import build_model
from utils import choose_device
from utils import ConfigParser
from utils.config_io import save_resolved_config
from utils.evaluator import Evaluator


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader,
    device: torch.device,
    amp: bool = False,
    threshold: float | None = 0.5,
) -> Dict[str, float]:
    """Run validation and return evaluator metrics."""
    del criterion

    model.eval()
    evaluator = Evaluator(threshold=threshold)
    use_amp = bool(amp) and device.type == "cuda"

    pbar = tqdm(dataloader, desc="Validate", leave=True, dynamic_ncols=True)
    for batch in pbar:
        prompt = batch["prompt"].to(device, non_blocking=True)
        query = batch["query"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(prompt, query)
        if logits.shape[1] >= 2:
            anomaly_scores = logits[:, 1:2]
        else:
            anomaly_scores = logits[:, 0:1]

        evaluator.update(
            preds=anomaly_scores,
            targets=mask,
            image_preds=None,
            image_gts=None,
            img_clsname=batch.get("img_clsname", None),
        )

    return evaluator.compute()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate VPAS checkpoint on val/test dataset.")
    parser.add_argument("--config", type=str, default="config/dinov3/vpas_dinov3_vitb16_test.yaml", help="Path to config file.")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "valid", "validation", "test"], help="Dataset split to validate.")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override validation batch size.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--threshold", type=float, default=None, help="Override evaluation threshold. Use a negative value to auto-search best F1 threshold.")
    parser.add_argument(
        "--json_path",
        type=str,
        nargs="+",
        action="append",
        default=None,
        help="Optional dataset json path(s). Supports repeated usage, e.g. --json_path a.json b.json --json_path c.json.",
    )
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save validation metrics as json.")
    return parser.parse_args()


def _resolve_threshold(cfg: ConfigParser, threshold_arg: float | None) -> float | None:
    if threshold_arg is None:
        # Binary classification score mode: do not use fixed threshold by default.
        return None
    if threshold_arg < 0:
        return None
    return float(threshold_arg)


def _load_model_from_ckpt(
    cfg: ConfigParser,
    ckpt_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    model, criterion = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    invalid_missing_keys = [k for k in missing_keys if not k.startswith("backbone.")]
    if invalid_missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint model state_dict mismatch. "
            f"Missing keys: {invalid_missing_keys}; unexpected keys: {unexpected_keys}"
        )
    model = model.to(device)
    criterion = criterion.to(device)
    model.eval()
    return model, criterion


def _resolve_eval_output_dir(args: argparse.Namespace) -> Path:
    if args.save_json:
        return Path(args.save_json).resolve().parent
    ckpt_dir = Path(args.ckpt).resolve().parent
    split_name = str(args.split).lower()
    return ckpt_dir / f"{split_name}_outputs"


def _flatten_json_paths(values: List[List[str]] | None) -> List[str]:
    if not values:
        return []
    flattened: List[str] = []
    for group in values:
        flattened.extend(str(item) for item in group)
    return flattened


def _collect_numeric_metrics(metrics_list: List[Dict[str, Any]]) -> List[str]:
    keys: set[str] = set()
    for metrics in metrics_list:
        for key, value in metrics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                keys.add(key)
    return sorted(keys)


def _compute_metric_statistics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    mean_metrics: Dict[str, float] = {}
    variance_metrics: Dict[str, float] = {}

    for key in _collect_numeric_metrics(metrics_list):
        values: List[float] = []
        for metrics in metrics_list:
            value = metrics.get(key, float("nan"))
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numeric = float(value)
                if not math.isnan(numeric):
                    values.append(numeric)

        if not values:
            continue

        count = float(len(values))
        mean_value = sum(values) / count
        variance_value = sum((value - mean_value) ** 2 for value in values) / count
        mean_metrics[key] = mean_value
        variance_metrics[key] = variance_value

    return {
        "mean": mean_metrics,
        "variance": variance_metrics,
    }


def _build_dataloader(cfg: ConfigParser, args: argparse.Namespace, device: torch.device) -> DataLoader:
    dataset = build_dataset(cfg, split=args.split, return_mask=True)

    batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg.train.get("batch_size", 1))
    num_workers = int(args.num_workers) if args.num_workers is not None else int(cfg.data.get("num_workers", 4))
    pin_memory = device.type == "cuda"
    persistent_workers = bool(cfg.data.get("persistent_workers", num_workers > 0)) and num_workers > 0
    prefetch_factor = int(cfg.data.get("prefetch_factor", 2)) if num_workers > 0 else None

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "persistent_workers": persistent_workers,
    }
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


def main() -> None:
    args = parse_args()
    cfg = ConfigParser.from_file(args.config)

    if args.device is not None:
        cfg.set("experiment.device", args.device)
    if args.batch_size is not None:
        cfg.set("train.batch_size", int(args.batch_size))

    eval_output_dir = _resolve_eval_output_dir(args)
    save_resolved_config(cfg, eval_output_dir / "resolved_config.yaml")

    device = choose_device(str(cfg.experiment.get("device", "cuda")))
    model, criterion = _load_model_from_ckpt(cfg, args.ckpt, device)

    split_name = str(args.split).lower()
    json_paths = [str(Path(path).expanduser().resolve()) for path in _flatten_json_paths(args.json_path)]
    threshold = _resolve_threshold(cfg, args.threshold)

    if json_paths:
        run_results: List[Dict[str, Any]] = []
        metrics_list: List[Dict[str, Any]] = []
        json_key = f"data.{split_name}_json"

        for idx, json_path in enumerate(json_paths, start=1):
            run_cfg = ConfigParser(cfg.to_dict())
            run_cfg.set(json_key, json_path)
            print(f"[val] ({idx}/{len(json_paths)}) validating json: {json_path}")
            dataloader = _build_dataloader(run_cfg, args, device)
            run_metrics = validate(
                model=model,
                criterion=criterion,
                dataloader=dataloader,
                device=device,
                amp=bool(run_cfg.train.get("amp", True)),
                threshold=threshold,
            )
            run_results.append(
                {
                    "json_path": json_path,
                    "metrics": run_metrics,
                }
            )
            metrics_list.append(run_metrics)

        metrics: Dict[str, Any] = {
            "split": split_name,
            "json_paths": json_paths,
            "runs": run_results,
            "aggregate": {
                "count": len(run_results),
                **_compute_metric_statistics(metrics_list),
            },
        }
    else:
        dataloader = _build_dataloader(cfg, args, device)
        metrics = validate(
            model=model,
            criterion=criterion,
            dataloader=dataloader,
            device=device,
            amp=bool(cfg.train.get("amp", True)),
            threshold=threshold,
        )

    metrics_json = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(metrics_json)

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(metrics_json + "\n", encoding="utf-8")
    else:
        save_path = eval_output_dir / "metrics.json"
        save_path.write_text(metrics_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
