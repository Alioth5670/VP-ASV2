import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ==============================
# Edit Config Here
# ==============================
CONFIG = {
    # "prompt" or "query"
    "base": "prompt",
    "prompt_dir": "dataset/rs19/images/rs19_val",
    "query_dir": "",
    "mask_dir": "",
    "output": "dataset/rs19/train.json",

    # Used only when base == "query"
    # Case A: multiple queries in same folder share one prompt file name
    # e.g., each scene folder has normal.png
    "shared_prompt_name": "normal.png",
    # Case B: global fallback prompt (absolute path or relative to prompt_dir)
    "default_prompt": None,

    # Common options
    "exts": (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
    "relative_path": False,
    "id_digits": 6,
}


def is_image_file(path: Path, exts: Sequence[str]) -> bool:
    return path.is_file() and path.suffix.lower() in exts


def collect_images(root: Path, exts: Sequence[str]) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    files = [p for p in root.rglob("*") if is_image_file(p, exts)]
    files.sort()
    print(f"[make_dataset_json] Found {len(files)} images in {root}")
    return files


def resolve_by_relative(base_dir: Optional[Path], rel_path: Path, exts: Sequence[str]) -> Optional[Path]:
    if base_dir is None:
        return None

    candidate = base_dir / rel_path
    if candidate.exists() and candidate.is_file():
        return candidate

    parent = candidate.parent
    stem = candidate.stem
    if parent.exists() and parent.is_dir():
        for ext in exts:
            alt = parent / f"{stem}{ext}"
            if alt.exists() and alt.is_file():
                return alt
    return None


def resolve_shared_name(
    prompt_dir: Path,
    rel_parent: Path,
    shared_prompt_name: str,
    exts: Sequence[str],
) -> Optional[Path]:
    target = prompt_dir / rel_parent / shared_prompt_name
    if target.suffix:
        return target if target.exists() and target.is_file() else None

    for ext in exts:
        p = target.with_suffix(ext)
        if p.exists() and p.is_file():
            return p
    return None


def resolve_default_prompt(prompt_dir: Path, default_prompt: Optional[str]) -> Optional[Path]:
    if not default_prompt:
        return None
    p = Path(default_prompt)
    if not p.is_absolute():
        p = prompt_dir / p
    return p if p.exists() and p.is_file() else None


def to_output_path(path: Path, absolute: bool) -> str:
    return str(path.resolve() if absolute else path)


def make_sample(
    idx: int,
    prompt_path: Path,
    query_path: Optional[Path],
    mask_path: Optional[Path],
    absolute_path: bool,
    id_digits: int,
) -> Dict[str, Optional[str]]:
    return {
        "id": str(idx).zfill(id_digits),
        "prompt_path": to_output_path(prompt_path, absolute_path),
        "query_path": to_output_path(query_path, absolute_path) if query_path is not None else None,
        "mask_path": to_output_path(mask_path, absolute_path) if mask_path is not None else None,
    }


def build_from_prompt(
    prompt_dir: Path,
    query_dir: Optional[Path],
    mask_dir: Optional[Path],
    exts: Sequence[str],
    absolute_path: bool,
    id_digits: int,
) -> List[Dict[str, Optional[str]]]:
    prompt_files = collect_images(prompt_dir, exts)
    if not prompt_files:
        raise RuntimeError(f"No images found in prompt_dir: {prompt_dir}")

    samples: List[Dict[str, Optional[str]]] = []
    for i, prompt_path in enumerate(prompt_files, start=1):
        rel = prompt_path.relative_to(prompt_dir)
        query_path = resolve_by_relative(query_dir, rel, exts)
        mask_path = resolve_by_relative(mask_dir, rel, exts)
        samples.append(make_sample(i, prompt_path, query_path, mask_path, absolute_path, id_digits))
    return samples


def build_from_query(
    prompt_dir: Path,
    query_dir: Path,
    mask_dir: Optional[Path],
    exts: Sequence[str],
    absolute_path: bool,
    id_digits: int,
    shared_prompt_name: Optional[str],
    default_prompt: Optional[str],
) -> List[Dict[str, Optional[str]]]:
    query_files = collect_images(query_dir, exts)
    if not query_files:
        raise RuntimeError(f"No images found in query_dir: {query_dir}")

    default_prompt_path = resolve_default_prompt(prompt_dir, default_prompt)
    if default_prompt and default_prompt_path is None:
        raise FileNotFoundError(f"default_prompt not found: {default_prompt}")

    samples: List[Dict[str, Optional[str]]] = []
    for i, query_path in enumerate(query_files, start=1):
        rel = query_path.relative_to(query_dir)

        prompt_path = resolve_by_relative(prompt_dir, rel, exts)
        if prompt_path is None and shared_prompt_name:
            prompt_path = resolve_shared_name(prompt_dir, rel.parent, shared_prompt_name, exts)
        if prompt_path is None and default_prompt_path is not None:
            prompt_path = default_prompt_path

        if prompt_path is None:
            raise FileNotFoundError(
                f"No prompt found for query: {query_path}. "
                f"Tried same-relative, shared_prompt_name, default_prompt."
            )

        mask_path = resolve_by_relative(mask_dir, rel, exts)
        samples.append(make_sample(i, prompt_path, query_path, mask_path, absolute_path, id_digits))

    return samples


def write_json(samples: List[Dict[str, Optional[str]]], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def print_stats(samples: List[Dict[str, Optional[str]]], output_json: Path, base: str) -> None:
    total = len(samples)
    query_hit = sum(1 for x in samples if x["query_path"] is not None)
    mask_hit = sum(1 for x in samples if x["mask_path"] is not None)
    print(f"[make_dataset_json] base: {base}")
    print(f"[make_dataset_json] total samples: {total}")
    print(f"[make_dataset_json] query matched: {query_hit}/{total}")
    print(f"[make_dataset_json] mask matched:  {mask_hit}/{total}")
    print(f"[make_dataset_json] saved to: {output_json.resolve()}")


def main() -> None:
    base = str(CONFIG["base"]).lower()
    if base not in {"prompt", "query"}:
        raise ValueError("CONFIG['base'] must be 'prompt' or 'query'")

    prompt_dir = Path(CONFIG["prompt_dir"])
    query_dir = Path(CONFIG["query_dir"]) if CONFIG.get("query_dir") else None
    mask_dir = Path(CONFIG["mask_dir"]) if CONFIG.get("mask_dir") else None
    output_json = Path(CONFIG["output"])

    exts = tuple(str(x).lower() for x in CONFIG.get("exts", (".png", ".jpg")))
    relative_path = bool(CONFIG.get("relative_path", False))
    absolute_path = not relative_path
    id_digits = int(CONFIG.get("id_digits", 6))

    if base == "prompt":
        samples = build_from_prompt(
            prompt_dir=prompt_dir,
            query_dir=query_dir,
            mask_dir=mask_dir,
            exts=exts,
            absolute_path=absolute_path,
            id_digits=id_digits,
        )
    else:
        if query_dir is None:
            raise ValueError("CONFIG['query_dir'] is required when base='query'")
        samples = build_from_query(
            prompt_dir=prompt_dir,
            query_dir=query_dir,
            mask_dir=mask_dir,
            exts=exts,
            absolute_path=absolute_path,
            id_digits=id_digits,
            shared_prompt_name=CONFIG.get("shared_prompt_name"),
            default_prompt=CONFIG.get("default_prompt"),
        )

    write_json(samples, output_json)
    print_stats(samples, output_json, base=base)


if __name__ == "__main__":
    main()
