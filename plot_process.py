#!/usr/bin/env python3
"""
plot_curves.py --datasetname <name>

For each (model, method) pair found in results/<datasetname>/:
  1. Plot cumulative accuracy curve as samples come in.
  2. Plot number of strategies (cheatsheet <memory_item> entries) as training continues.

Saves figures to figures/<datasetname>_accuracy_curve.png
                  figures/<datasetname>_strategy_count.png
"""

import argparse
import json
import re
from pathlib import Path

import os
os.environ.pop('MPLBACKEND', None)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Evaluation helpers (mirrors summarize.py)
# ---------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(r"_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}_?$")


def _extract_letter(text: str) -> str:
    m = re.search(r"\(([A-F])\)", text, re.IGNORECASE)
    if m:
        return f"({m.group(1).upper()})"
    for letter in "ABCDEF":
        if letter in text.upper():
            return f"({letter})"
    return text.strip()


def _clean_bound(expr: str) -> str:
    expr = expr.replace("$", "").replace("\\boxed{", "").replace("}", "")
    expr = re.sub(r"\s+", "", expr)
    if expr.lower().startswith("c="):
        expr = expr[2:]
    return expr.strip()


def is_correct(record: dict) -> bool:
    final = str(record.get("final_answer", "")).strip()
    target = str(record.get("target", "")).strip()
    problem_type = record.get("problem_type", "")
    if problem_type == "relation":
        return _extract_letter(final) == _extract_letter(target)
    if problem_type == "bound":
        return _clean_bound(final) == _clean_bound(target)
    return final == target


def count_strategies(record: dict) -> int:
    """Count <memory_item> entries in the final_cheatsheet."""
    cs = record.get("final_cheatsheet", "") or ""
    return len(re.findall(r"<memory_item>", cs))


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _parse_stem(stem: str) -> tuple[str, str] | None:
    stem = _TIMESTAMP_RE.sub("", stem)
    parts = stem.split("_", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def load_all_records(results_dir: Path) -> dict[tuple[str, str], list[dict]]:
    """
    Returns: {(model, method): [records in order]}
    When multiple files map to the same (model, method), the one with the
    most records (or latest name) wins.
    """
    buckets: dict[tuple[str, str], list[dict]] = {}

    for jsonl_file in sorted(results_dir.glob("*.jsonl")):
        parsed = _parse_stem(jsonl_file.stem)
        if parsed is None:
            print(f"Skipping unrecognized filename: {jsonl_file.name}")
            continue
        model, method = parsed

        records: list[dict] = []
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            continue

        key = (model, method)
        if key not in buckets or len(records) >= len(buckets[key]):
            buckets[key] = records

    return buckets


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _method_sort_key(method: str) -> int:
    lower = method.lower()
    if "strategic" in lower:
        return 5
    if "synthesis" in lower:
        return 4
    if "retrieval" in lower or "retrieve" in lower:
        return 3
    if "cumulative" in lower:
        return 2
    if "empty" in lower:
        return 1
    if "baseline" in lower or "default" in lower:
        return 0
    return 6


def _cumulative_accuracy(records: list[dict]) -> np.ndarray:
    correct = np.array([is_correct(r) for r in records], dtype=float)
    return np.cumsum(correct) / np.arange(1, len(correct) + 1)


def _strategy_counts(records: list[dict]) -> np.ndarray:
    return np.array([count_strategies(r) for r in records])


# ---------------------------------------------------------------------------
# Main plot functions
# ---------------------------------------------------------------------------

def plot_accuracy_curve(
    dataset_name: str,
    buckets: dict[tuple[str, str], list[dict]],
) -> None:
    """Plot cumulative accuracy over samples for each (model, method)."""
    # Group by model so we can use different line styles per model
    models = sorted({m for m, _ in buckets})
    methods = sorted({meth for _, meth in buckets}, key=_method_sort_key)

    cmap = plt.get_cmap("tab10")
    method_colors = {meth: cmap(i % 10) for i, meth in enumerate(methods)}
    model_styles = {m: ls for m, ls in zip(models, ["-", "--", ":", "-."])}

    fig, ax = plt.subplots(figsize=(10, 5))

    for (model, method), records in sorted(buckets.items(), key=lambda kv: (_method_sort_key(kv[0][1]), kv[0][0])):
        acc = _cumulative_accuracy(records)
        xs = np.arange(1, len(acc) + 1)
        label = f"{model} / {method}"
        ax.plot(
            xs, acc,
            color=method_colors[method],
            linestyle=model_styles.get(model, "-"),
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel("Samples seen")
    ax.set_ylabel("Cumulative accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Accuracy curve — {dataset_name}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    safe_name = dataset_name.replace("/", "_")
    out_path = Path("figures") / f"{safe_name}_accuracy_curve.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_strategy_count(
    dataset_name: str,
    buckets: dict[tuple[str, str], list[dict]],
) -> None:
    """Plot number of strategies in cheatsheet over samples for each (model, method)."""
    # Only include entries that actually have cheatsheet data
    cheatsheet_buckets = {
        (model, method): records
        for (model, method), records in buckets.items()
        if any(count_strategies(r) > 0 for r in records)
    }

    if not cheatsheet_buckets:
        print("No cheatsheet strategy data found — skipping strategy count plot.")
        return

    models = sorted({m for m, _ in cheatsheet_buckets})
    methods = sorted({meth for _, meth in cheatsheet_buckets}, key=_method_sort_key)

    cmap = plt.get_cmap("tab10")
    method_colors = {meth: cmap(i % 10) for i, meth in enumerate(methods)}
    model_styles = {m: ls for m, ls in zip(models, ["-", "--", ":", "-."])}

    fig, ax = plt.subplots(figsize=(10, 5))

    for (model, method), records in sorted(
        cheatsheet_buckets.items(),
        key=lambda kv: (_method_sort_key(kv[0][1]), kv[0][0]),
    ):
        counts = _strategy_counts(records)
        xs = np.arange(1, len(counts) + 1)
        label = f"{model} / {method}"
        ax.plot(
            xs, counts,
            color=method_colors[method],
            linestyle=model_styles.get(model, "-"),
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel("Samples seen")
    ax.set_ylabel("Number of strategies in cheatsheet")
    ax.set_title(f"Strategy count — {dataset_name}")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    safe_name = dataset_name.replace("/", "_")
    out_path = Path("figures") / f"{safe_name}_strategy_count.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot accuracy curves and strategy counts.")
    parser.add_argument("--datasetname", required=True, help="Dataset subdirectory under results/")
    args = parser.parse_args()

    results_dir = Path("results") / args.datasetname
    if not results_dir.is_dir():
        raise SystemExit(f"Directory not found: {results_dir}")

    buckets = load_all_records(results_dir)
    if not buckets:
        raise SystemExit("No results loaded — check that the directory contains .jsonl files.")

    print(f"Loaded {len(buckets)} (model, method) pairs.")

    plot_accuracy_curve(args.datasetname, buckets)
    plot_strategy_count(args.datasetname, buckets)


if __name__ == "__main__":
    main()
