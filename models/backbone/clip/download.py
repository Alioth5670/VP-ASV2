from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from .config import get_clip_model_url, get_default_checkpoint_path, is_clip_model_name


SUPPORTED_MODELS = (
    "CLIP_ViT-B/32",
    "CLIP_ViT-B/16",
    "CLIP_ViT-L/14",
    "CLIP_ViT-L/14@336px",
)


def expected_sha256_from_url(url: str) -> str:
    parts = url.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Unexpected CLIP url format: {url}")
    return parts[-2]


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp_handle:
        tmp_path = Path(tmp_handle.name)

    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out_handle:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out_handle.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = downloaded * 100.0 / total
                    print(f"Downloaded {downloaded}/{total} bytes ({percent:.1f}%)", end="\r", flush=True)
        if total > 0:
            print()
        os.replace(tmp_path, destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def download_clip_model(model_name: str, output: str | Path | None = None) -> Path:
    if not is_clip_model_name(model_name):
        raise ValueError(f"Unknown CLIP model: {model_name}")

    url = get_clip_model_url(model_name)
    destination = Path(output) if output is not None else get_default_checkpoint_path(model_name)
    expected_sha = expected_sha256_from_url(url)

    if destination.is_file():
        actual_sha = sha256sum(destination)
        if actual_sha == expected_sha:
            print(f"Using existing checkpoint: {destination}")
            return destination
        print(f"Checksum mismatch for existing file, re-downloading: {destination}")

    print(f"Downloading {model_name} from {url}")
    download_with_progress(url, destination)

    actual_sha = sha256sum(destination)
    if actual_sha != expected_sha:
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {destination}. Expected {expected_sha}, got {actual_sha}."
        )

    print(f"Saved checkpoint to {destination}")
    return destination


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download OpenAI CLIP vision checkpoints.")
    parser.add_argument("model_name", type=str, help=f"One of: {', '.join(SUPPORTED_MODELS)}")
    parser.add_argument("--output", type=str, default=None, help="Optional output checkpoint path.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    download_clip_model(args.model_name, output=args.output)


if __name__ == "__main__":
    main()
