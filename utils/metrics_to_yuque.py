from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

IMAGE_METRICS = [
    ("AUROC", "image_auroc"),
    ("AUPR", "image_aupr"),
    ("F1", "image_f1"),
    ("AP", "image_ap"),
]

PIXEL_METRICS = [
    ("AUROC", "pixel_auroc"),
    ("AUPR", "pixel_aupr"),
    ("F1", "pixel_f1"),
    ("AP", "pixel_ap"),
]

METRIC_GROUPS = [
    ("image", IMAGE_METRICS),
    ("pixel", PIXEL_METRICS),
]

VARIANCE_KEY = "__variance__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert metrics.json files into Yuque-friendly HTML tables."
    )
    parser.add_argument(
        "metrics_json",
        nargs="*",
        type=Path,
        help="Optional metrics.json paths. If omitted, only outputs containing 'noite' are scanned.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Outputs directory used when auto-scanning. Default: outputs",
    )
    parser.add_argument(
        "--keyword",
        default="noite",
        help="Only auto-scan experiment folders whose names contain this keyword. Default: noite",
    )
    parser.add_argument(
        "--metrics-subdir",
        default=None,
        help="Optional metrics output subdirectory to select, e.g. test_outputs or test_outputs_best.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        help="Optional model names matching the metrics.json inputs.",
    )
    parser.add_argument(
        "--split-metric-tables",
        action="store_true",
        help="Output one summary table per metric instead of the grouped matrix table.",
    )
    parser.add_argument(
        "--mean-std",
        action="store_true",
        help="Show values as mean±std when aggregate variance is available.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output html path. Prints to stdout when omitted.",
    )
    return parser.parse_args()



def load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data



def infer_model_name(path: Path) -> str:
    return path.parent.parent.name



def discover_metrics(outputs_dir: Path, keyword: str, metrics_subdir: str | None = None) -> list[Path]:
    base_dir = outputs_dir.expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {base_dir}")

    discovered: list[Path] = []
    lowered = keyword.lower()
    for experiment_dir in sorted(base_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue
        if lowered not in experiment_dir.name.lower():
            continue
        if metrics_subdir:
            metrics_path = (experiment_dir / metrics_subdir / "metrics.json").resolve()
            if metrics_path.exists():
                discovered.append(metrics_path)
            continue
        for metrics_path in sorted(experiment_dir.glob("**/metrics.json")):
            discovered.append(metrics_path.resolve())

    if not discovered:
        suffix = f" in subdirectory '{metrics_subdir}'" if metrics_subdir else ""
        raise FileNotFoundError(
            f"No metrics.json found under {base_dir} for experiment folders containing '{keyword}'{suffix}."
        )
    return discovered



def resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    if args.metrics_json:
        return [path.expanduser().resolve() for path in args.metrics_json]
    return discover_metrics(args.outputs_dir, args.keyword, args.metrics_subdir)



def normalize_models(paths: list[Path], names: list[str] | None) -> list[tuple[str, Path, dict[str, Any]]]:
    if names and len(names) != len(paths):
        raise ValueError("--names count must match the number of metrics.json inputs.")

    models = []
    for index, path in enumerate(paths):
        name = names[index] if names else infer_model_name(path)
        models.append((name, path, load_metrics(path)))
    return models



def get_summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    aggregate = metrics.get("aggregate", {})
    if isinstance(aggregate, dict):
        mean_metrics = aggregate.get("mean", {})
        variance_metrics = aggregate.get("variance", {})
        if isinstance(mean_metrics, dict) and mean_metrics:
            row_metrics = dict(mean_metrics)
            if isinstance(variance_metrics, dict) and variance_metrics:
                row_metrics[VARIANCE_KEY] = variance_metrics
            return row_metrics

    average = metrics.get("average", {})
    if isinstance(average, dict) and average:
        return average

    summary = {}
    for key, value in metrics.items():
        if key in {"per_class", "average", "aggregate", "runs", "json_paths", "split"}:
            continue
        summary[key] = value
    return summary



def format_percent(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"

    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return "-"

    return f"{numeric * 100:.2f}"



def format_percent_with_std(value: Any, variance: Any, show_mean_std: bool) -> str:
    mean_text = format_percent(value)
    if mean_text == "-" or not show_mean_std:
        return mean_text

    if not isinstance(variance, (int, float)) or isinstance(variance, bool):
        return mean_text

    variance_value = float(variance)
    if math.isnan(variance_value) or math.isinf(variance_value) or variance_value < 0:
        return mean_text

    std_text = format_percent(math.sqrt(variance_value))
    if std_text == "-":
        return mean_text
    return f"{mean_text}±{std_text}"



def format_metric_value(row_metrics: dict[str, Any], metric_key: str, show_mean_std: bool) -> str:
    variance_metrics = row_metrics.get(VARIANCE_KEY, {})
    if not isinstance(variance_metrics, dict):
        variance_metrics = {}
    return format_percent_with_std(row_metrics.get(metric_key), variance_metrics.get(metric_key), show_mean_std)



def build_grouped_summary_table(models: list[tuple[str, Path, dict[str, Any]]], show_mean_std: bool) -> str:
    lines = [
        "<table>",
        "  <thead>",
        "    <tr>",
        '      <th colspan="2">class</th>',
    ]

    for model_name, _, _ in models:
        lines.append(f"      <th>{model_name}</th>")

    lines.extend([
        "    </tr>",
        "  </thead>",
        "  <tbody>",
    ])

    for group_name, metric_defs in METRIC_GROUPS:
        for metric_index, (metric_label, metric_key) in enumerate(metric_defs):
            lines.append("    <tr>")
            if metric_index == 0:
                lines.append(f'      <td rowspan="{len(metric_defs)}">{group_name}</td>')
            lines.append(f"      <td>{metric_label}</td>")
            for _, _, metrics in models:
                summary_metrics = get_summary_metrics(metrics)
                lines.append(f"      <td>{format_metric_value(summary_metrics, metric_key, show_mean_std)}</td>")
            lines.append("    </tr>")

    lines.extend([
        "  </tbody>",
        "</table>",
    ])
    return "\n".join(lines)



def build_single_metric_table(
    models: list[tuple[str, Path, dict[str, Any]]],
    group_name: str,
    metric_label: str,
    metric_key: str,
    show_mean_std: bool,
) -> str:
    lines = [
        f"<h3>{group_name} {metric_label}</h3>",
        "<table>",
        "  <thead>",
        "    <tr>",
        "      <th>class</th>",
    ]

    for model_name, _, _ in models:
        lines.append(f"      <th>{model_name}</th>")

    lines.extend([
        "    </tr>",
        "  </thead>",
        "  <tbody>",
        "    <tr>",
        "      <td>average</td>",
    ])

    for _, _, metrics in models:
        summary_metrics = get_summary_metrics(metrics)
        lines.append(f"      <td>{format_metric_value(summary_metrics, metric_key, show_mean_std)}</td>")

    lines.extend([
        "    </tr>",
        "  </tbody>",
        "</table>",
    ])
    return "\n".join(lines)



def build_split_metric_tables(models: list[tuple[str, Path, dict[str, Any]]], show_mean_std: bool) -> str:
    sections = []
    for group_name, metric_defs in METRIC_GROUPS:
        for metric_label, metric_key in metric_defs:
            sections.append(build_single_metric_table(models, group_name, metric_label, metric_key, show_mean_std))
    return "\n\n".join(sections)



def build_output(models: list[tuple[str, Path, dict[str, Any]]], split_metric_tables: bool, show_mean_std: bool) -> str:
    if split_metric_tables:
        return build_split_metric_tables(models, show_mean_std) + "\n"
    return build_grouped_summary_table(models, show_mean_std) + "\n"



def main() -> None:
    args = parse_args()
    paths = resolve_input_paths(args)
    models = normalize_models(paths, args.names)
    html = build_output(models, args.split_metric_tables, args.mean_std)

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        print(f"Yuque table written to: {output_path}")
        return

    print(html, end="")


if __name__ == "__main__":
    main()
