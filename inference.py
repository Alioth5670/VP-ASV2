from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from torchvision.transforms import functional as TF

from data.vpas_dataset import _is_missing_path, load_json_list
from models.vpas import build_model
from utils import ConfigParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for VPAS on image pairs or video.")
    parser.add_argument("--config", type=str, default="config/vpas.yaml", help="Path to config file.")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path, e.g. best.pth or last.pth.")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--mode", type=str, required=True, choices=["image", "video"], help="Inference mode.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save outputs.")

    parser.add_argument("--prompt-image", type=str, default=None, help="Prompt image path for image mode, or optional prompt image for video mode.")
    parser.add_argument("--query-image", type=str, default=None, help="Query image path for image mode.")
    parser.add_argument("--query-dir", type=str, default=None, help="Directory of query images for image mode.")
    parser.add_argument("--input-json", type=str, default=None, help="Dataset-style JSON file for batch image inference.")

    parser.add_argument("--video-path", type=str, default=None, help="Input video path for video mode.")
    parser.add_argument(
        "--video-prompt-mode",
        type=str,
        default="first_n_mean",
        choices=["first_frame", "first_n_mean", "all_mean"],
        help="How to build the prompt image from the video.",
    )
    parser.add_argument(
        "--video-prompt-frames",
        type=int,
        default=8,
        help="Number of leading frames used when --video-prompt-mode=first_n_mean.",
    )
    parser.add_argument("--video-max-frames", type=int, default=-1, help="Optional max number of frames to process; -1 means all.")
    parser.add_argument("--video-stride", type=int, default=1, help="Process every N-th frame in video mode.")
    parser.add_argument("--save-mask-video", action="store_true", help="Also save binary mask video in video mode.")
    parser.add_argument("--save-heatmap-video", action="store_true", help="Also save heatmap video in video mode.")
    parser.add_argument(
        "--export-frame-indices",
        type=int,
        nargs="*",
        default=None,
        help="Specific processed-frame indices to export as images in video mode.",
    )
    parser.add_argument(
        "--export-frame-every",
        type=int,
        default=0,
        help="Export one result image every N processed frames in video mode; 0 disables it.",
    )
    return parser.parse_args()


def choose_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        if args.mode == "image":
            if args.input_json:
                stem = Path(args.input_json).stem
            elif args.query_image:
                stem = Path(args.query_image).stem
            elif args.query_dir:
                stem = Path(args.query_dir).stem
            else:
                stem = "image"
        else:
            stem = Path(args.video_path).stem
        out_dir = Path("outputs") / "inference" / f"{args.mode}_{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_checkpoint_model(cfg: ConfigParser, ckpt_path: str | Path, device: torch.device) -> torch.nn.Module:
    model, _ = build_model(cfg)
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
    model.eval()
    return model


def load_rgb_pil(path: str | Path) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    with Image.open(p) as img:
        return img.convert("RGB")


def list_query_images(query_dir: str | Path) -> list[Path]:
    root = Path(query_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Query directory not found: {root}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not files:
        raise ValueError(f"No image files found in query directory: {root}")
    return files


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def resolve_json_asset_path(path_value: str | Path, json_path: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    candidate_from_json = (json_path.parent / candidate).resolve()
    if candidate_from_json.exists():
        return candidate_from_json
    return candidate.resolve()


def load_pair_json_samples(json_path: str | Path) -> list[dict[str, Any]]:
    json_file = Path(json_path).expanduser().resolve()
    raw_samples = load_json_list(json_file)

    flattened: list[dict[str, Any]] = []
    for item_idx, item in enumerate(raw_samples):
        prompt_path_raw = item.get("prompt_path")
        if _is_missing_path(prompt_path_raw):
            raise ValueError(f"Missing prompt_path in item index {item_idx}.")

        prompt_path = resolve_json_asset_path(prompt_path_raw, json_file)
        query_paths = ensure_list(item.get("query_path"))
        if not query_paths:
            raise ValueError(f"Missing query_path in item index {item_idx}.")

        for query_idx, query_path_raw in enumerate(query_paths):
            if _is_missing_path(query_path_raw):
                raise ValueError(f"Invalid query_path in item index {item_idx}, query index {query_idx}.")
            query_path = resolve_json_asset_path(query_path_raw, json_file)
            flattened.append(
                {
                    "img_clsname": item.get("img_clsname"),
                    "prompt_path": prompt_path,
                    "query_path": query_path,
                    "query_index": query_idx,
                }
            )
    return flattened


def build_json_sample_stem(sample: dict[str, Any]) -> str:
    query_path = Path(sample["query_path"])
    tail_parts = list(query_path.parts[-3:])
    if tail_parts:
        tail_parts[-1] = Path(tail_parts[-1]).stem
    stem = "__".join(tail_parts) if tail_parts else query_path.stem
    if sample.get("query_index", 0):
        stem = f"{stem}__q{int(sample['query_index'])}"
    return sanitize_name(stem)


def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def get_infer_size_and_norm(cfg: ConfigParser) -> tuple[Tuple[int, int], bool, list[float], list[float], float, bool]:
    data_cfg = cfg.data
    output_size = data_cfg.get("val_output_size", data_cfg.get("output_size", [512, 512]))
    if isinstance(output_size, int):
        size_hw = (int(output_size), int(output_size))
    else:
        size_hw = (int(output_size[0]), int(output_size[1]))

    normalize_cfg = data_cfg.get("normalize", False)
    if isinstance(normalize_cfg, dict):
        normalize = bool(normalize_cfg.get("enabled", True))
        mean = [float(v) for v in normalize_cfg.get("mean", [0.485, 0.456, 0.406])]
        std = [float(v) for v in normalize_cfg.get("std", [0.229, 0.224, 0.225])]
    else:
        normalize = bool(normalize_cfg)
        mean = [float(v) for v in data_cfg.get("mean", [0.485, 0.456, 0.406])]
        std = [float(v) for v in data_cfg.get("std", [0.229, 0.224, 0.225])]

    threshold = float(cfg.inference.get("threshold", 0.5))
    amp = bool(cfg.train.get("amp", True))
    return size_hw, normalize, mean, std, threshold, amp


def preprocess_image(
    img: Image.Image,
    size_hw: Tuple[int, int],
    normalize: bool,
    mean: Iterable[float],
    std: Iterable[float],
) -> torch.Tensor:
    img = img.convert("RGB")
    target_h, target_w = size_hw
    if img.size != (target_w, target_h):
        img = TF.resize(img, (target_h, target_w), interpolation=InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(img)
    if normalize:
        tensor = TF.normalize(tensor, mean=list(mean), std=list(std))
    return tensor.unsqueeze(0)


@torch.no_grad()
def infer_pair(
    model: torch.nn.Module,
    prompt_img: Image.Image,
    query_img: Image.Image,
    size_hw: Tuple[int, int],
    normalize: bool,
    mean: Iterable[float],
    std: Iterable[float],
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    prompt_t = preprocess_image(prompt_img, size_hw, normalize, mean, std).to(device)
    query_t = preprocess_image(query_img, size_hw, normalize, mean, std).to(device)
    use_amp = bool(amp) and device.type == "cuda"
    with torch.autocast(device_type=device.type, enabled=use_amp):
        logits = model(prompt_t, query_t)
    if logits.shape[1] >= 2:
        anomaly_score = logits[0, 1].detach().cpu().numpy().astype(np.float32)
        pred_cls = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        binary_mask = pred_cls * 255
    else:
        anomaly_score = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
        binary_mask = None
    return anomaly_score, binary_mask


def resize_prob_to_image(prob_map: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(prob_map, dsize=size_wh, interpolation=cv2.INTER_LINEAR)


def to_binary_mask(prob_map: np.ndarray, threshold: float) -> np.ndarray:
    return ((prob_map >= threshold).astype(np.uint8) * 255)


def to_heatmap(prob_map: np.ndarray) -> np.ndarray:
    score = np.asarray(prob_map, dtype=np.float32)
    finite = np.isfinite(score)
    if not finite.any():
        heat = np.zeros_like(score, dtype=np.uint8)
        return cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    score_min = float(score[finite].min())
    score_max = float(score[finite].max())
    denom = max(score_max - score_min, 1e-6)
    normalized = np.clip((score - score_min) / denom, 0.0, 1.0)
    heat = (normalized * 255.0).astype(np.uint8)
    return cv2.applyColorMap(heat, cv2.COLORMAP_JET)


def make_overlay(base_bgr: np.ndarray, prob_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = to_heatmap(prob_map)
    return cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap, alpha, 0.0)


def save_image_outputs(
    out_dir: Path,
    prompt_img: Image.Image,
    query_img: Image.Image,
    prob_map: np.ndarray,
    threshold: float,
    binary_mask: np.ndarray | None = None,
    stem: str | None = None,
) -> None:
    prompt_bgr = pil_to_bgr(prompt_img)
    query_bgr = pil_to_bgr(query_img)
    prob_resized = resize_prob_to_image(prob_map, (query_bgr.shape[1], query_bgr.shape[0]))
    if binary_mask is not None:
        mask = cv2.resize(binary_mask, dsize=(query_bgr.shape[1], query_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = to_binary_mask(prob_resized, threshold)
    heatmap = to_heatmap(prob_resized)
    overlay = make_overlay(query_bgr, prob_resized)

    prefix = "" if stem is None else f"{stem}_"
    cv2.imwrite(str(out_dir / f"{prefix}prompt.png"), prompt_bgr)
    cv2.imwrite(str(out_dir / f"{prefix}query.png"), query_bgr)
    cv2.imwrite(str(out_dir / f"{prefix}mask.png"), mask)
    cv2.imwrite(str(out_dir / f"{prefix}heatmap.png"), heatmap)
    cv2.imwrite(str(out_dir / f"{prefix}overlay.png"), overlay)
    np.save(out_dir / f"{prefix}prob.npy", prob_resized)


def save_video_frame_outputs(
    out_dir: Path,
    frame_idx: int,
    prompt_img: Image.Image,
    query_bgr: np.ndarray,
    prob_resized: np.ndarray,
    threshold: float,
    binary_mask_resized: np.ndarray | None = None,
) -> None:
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    prompt_bgr = pil_to_bgr(prompt_img)
    mask = binary_mask_resized if binary_mask_resized is not None else to_binary_mask(prob_resized, threshold)
    heatmap = to_heatmap(prob_resized)
    overlay = make_overlay(query_bgr, prob_resized)

    cv2.imwrite(str(frame_dir / f"frame_{frame_idx:06d}_prompt.png"), prompt_bgr)
    cv2.imwrite(str(frame_dir / f"frame_{frame_idx:06d}_query.png"), query_bgr)
    cv2.imwrite(str(frame_dir / f"frame_{frame_idx:06d}_mask.png"), mask)
    cv2.imwrite(str(frame_dir / f"frame_{frame_idx:06d}_heatmap.png"), heatmap)
    cv2.imwrite(str(frame_dir / f"frame_{frame_idx:06d}_overlay.png"), overlay)
    np.save(frame_dir / f"frame_{frame_idx:06d}_prob.npy", prob_resized)


def read_video_frames(video_path: str | Path, max_frames: int = -1, stride: int = 1) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 25.0

    frames: list[np.ndarray] = []
    frame_idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if stride > 1 and frame_idx % stride != 0:
            frame_idx += 1
            continue
        frames.append(frame)
        kept += 1
        frame_idx += 1
        if max_frames > 0 and kept >= max_frames:
            break
    cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from video: {video_path}")
    return frames, fps


def average_frames(frames: list[np.ndarray]) -> np.ndarray:
    acc = np.zeros_like(frames[0], dtype=np.float64)
    for frame in frames:
        acc += frame.astype(np.float64)
    avg = np.clip(acc / max(len(frames), 1), 0.0, 255.0).astype(np.uint8)
    return avg


def build_video_prompt(frames: list[np.ndarray], mode: str, first_n: int) -> np.ndarray:
    if mode == "first_frame":
        return frames[0]
    if mode == "first_n_mean":
        return average_frames(frames[: max(first_n, 1)])
    if mode == "all_mean":
        return average_frames(frames)
    raise ValueError(f"Unsupported video prompt mode: {mode}")


def run_image_mode(args: argparse.Namespace, cfg: ConfigParser, model: torch.nn.Module, device: torch.device, out_dir: Path) -> None:
    size_hw, normalize, mean, std, threshold, amp = get_infer_size_and_norm(cfg)

    if args.input_json:
        if args.prompt_image or args.query_image or args.query_dir:
            raise ValueError("When --input-json is provided, do not also pass --prompt-image/--query-image/--query-dir.")

        samples = load_pair_json_samples(args.input_json)
        pbar = tqdm(samples, desc="Infer JSON Samples", dynamic_ncols=True)
        for sample in pbar:
            prompt_img = load_rgb_pil(sample["prompt_path"])
            query_img = load_rgb_pil(sample["query_path"])
            prob_map, binary_mask = infer_pair(model, prompt_img, query_img, size_hw, normalize, mean, std, device, amp)

            class_dir = out_dir / sanitize_name(str(sample.get("img_clsname") or "unclassified"))
            class_dir.mkdir(parents=True, exist_ok=True)
            save_image_outputs(
                class_dir,
                prompt_img,
                query_img,
                prob_map,
                threshold,
                binary_mask=binary_mask,
                stem=build_json_sample_stem(sample),
            )

        print(f"[done] saved json image inference outputs to {out_dir.resolve()}")
        return

    if not args.prompt_image:
        raise ValueError("Image mode requires --prompt-image.")
    if not args.query_image and not args.query_dir:
        raise ValueError("Image mode requires either --query-image or --query-dir.")

    prompt_img = load_rgb_pil(args.prompt_image)

    if args.query_dir:
        query_paths = list_query_images(args.query_dir)
        pbar = tqdm(query_paths, desc="Infer Images", dynamic_ncols=True)
        for query_path in pbar:
            query_img = load_rgb_pil(query_path)
            prob_map, binary_mask = infer_pair(model, prompt_img, query_img, size_hw, normalize, mean, std, device, amp)
            save_image_outputs(
                out_dir,
                prompt_img,
                query_img,
                prob_map,
                threshold,
                binary_mask=binary_mask,
                stem=query_path.stem,
            )
        print(f"[done] saved folder image inference outputs to {out_dir.resolve()}")
        return

    query_img = load_rgb_pil(args.query_image)
    prob_map, binary_mask = infer_pair(model, prompt_img, query_img, size_hw, normalize, mean, std, device, amp)
    save_image_outputs(out_dir, prompt_img, query_img, prob_map, threshold, binary_mask=binary_mask)
    print(f"[done] saved image inference outputs to {out_dir.resolve()}")


def run_video_mode(args: argparse.Namespace, cfg: ConfigParser, model: torch.nn.Module, device: torch.device, out_dir: Path) -> None:
    if not args.video_path:
        raise ValueError("Video mode requires --video-path.")

    size_hw, normalize, mean, std, threshold, amp = get_infer_size_and_norm(cfg)
    frames, fps = read_video_frames(args.video_path, max_frames=args.video_max_frames, stride=max(args.video_stride, 1))

    if args.prompt_image:
        prompt_img = load_rgb_pil(args.prompt_image)
        cv2.imwrite(str(out_dir / "prompt_input.png"), pil_to_bgr(prompt_img))
    else:
        prompt_frame = build_video_prompt(frames, args.video_prompt_mode, args.video_prompt_frames)
        prompt_img = bgr_to_pil(prompt_frame)
        cv2.imwrite(str(out_dir / "prompt_from_video.png"), prompt_frame)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(str(out_dir / "overlay.mp4"), fourcc, fps, (w, h))
    heatmap_writer = None
    mask_writer = None
    if args.save_heatmap_video:
        heatmap_writer = cv2.VideoWriter(str(out_dir / "heatmap.mp4"), fourcc, fps, (w, h))
    if args.save_mask_video:
        mask_writer = cv2.VideoWriter(str(out_dir / "mask.mp4"), fourcc, fps, (w, h), isColor=False)

    export_indices = set(args.export_frame_indices or [])
    export_every = max(int(args.export_frame_every), 0)

    pbar = tqdm(enumerate(frames), total=len(frames), desc="Infer Video", dynamic_ncols=True)
    for idx, frame_bgr in pbar:
        query_img = bgr_to_pil(frame_bgr)
        prob_map, binary_mask = infer_pair(model, prompt_img, query_img, size_hw, normalize, mean, std, device, amp)
        prob_resized = resize_prob_to_image(prob_map, (w, h))
        binary_mask_resized = None
        if binary_mask is not None:
            binary_mask_resized = cv2.resize(binary_mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        overlay = make_overlay(frame_bgr, prob_resized)
        overlay_writer.write(overlay)

        if heatmap_writer is not None:
            heatmap_writer.write(to_heatmap(prob_resized))
        if mask_writer is not None:
            if binary_mask_resized is not None:
                mask_writer.write(binary_mask_resized)
            else:
                mask_writer.write(to_binary_mask(prob_resized, threshold))

        should_export = idx in export_indices or (export_every > 0 and idx % export_every == 0)
        if should_export:
            save_video_frame_outputs(
                out_dir,
                idx,
                prompt_img,
                frame_bgr,
                prob_resized,
                threshold,
                binary_mask_resized=binary_mask_resized,
            )

    overlay_writer.release()
    if heatmap_writer is not None:
        heatmap_writer.release()
    if mask_writer is not None:
        mask_writer.release()

    print(f"[done] saved video inference outputs to {out_dir.resolve()}")


def main() -> None:
    args = parse_args()
    cfg = ConfigParser.from_file(args.config)
    device = choose_device(args.device or str(cfg.experiment.get("device", "cuda")))
    out_dir = resolve_output_dir(args)
    model = load_checkpoint_model(cfg, args.ckpt, device)

    if args.mode == "image":
        run_image_mode(args, cfg, model, device, out_dir)
    elif args.mode == "video":
        run_video_mode(args, cfg, model, device, out_dir)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
