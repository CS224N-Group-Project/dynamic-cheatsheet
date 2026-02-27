#!/usr/bin/env python3
"""
summarize.py --datasetname <name>
Plot accuracy results from results/<datasetname>/ directory.
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
# Evaluation helpers (mirrors dynamic_cheatsheet/utils/evaluation.py)
# ---------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(r"_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}_?$")


def _extract_letter(text: str) -> str:
    """Return the first (A)-(F) letter found in text, else the bare letter."""
    m = re.search(r"\(([A-F])\)", text, re.IGNORECASE)
    if m:
        return f"({m.group(1).upper()})"
    for letter in "ABCDEF":
        if letter in text.upper():
            return f"({letter})"
    return text.strip()


def _clean_bound(expr: str) -> str:
    """Normalize a bound expression for comparison."""
    expr = expr.replace("$", "").replace("\\boxed{", "").replace("}", "")
    expr = re.sub(r"\s+", "", expr)
    if expr.lower().startswith("c="):
        expr = expr[2:]
    return expr.strip()


def is_correct(record: dict) -> bool:
    """Return True if the record's final_answer matches its target."""
    final = str(record.get("final_answer", "")).strip()
    target = str(record.get("target", "")).strip()
    problem_type = record.get("problem_type", "")

    if problem_type == "relation":
        return _extract_letter(final) == _extract_letter(target)

    if problem_type == "bound":
        return _clean_bound(final) == _clean_bound(target)

    # Default: exact match (works for AIME, GPQA, MMLU, etc.)
    return final == target


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _parse_stem(stem: str) -> tuple[str, str] | None:
    """
    Parse '{model}_{method}[_{timestamp}]' into (model, method).
    Strips trailing timestamp if present (format YYYY-MM-DD-HH-MM).
    Returns None if the stem cannot be parsed.
    """
    stem = _TIMESTAMP_RE.sub("", stem)  # strip timestamp suffix
    parts = stem.split("_", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def load_results(results_dir: Path) -> dict[str, dict[str, float]]:
    """
    Returns: {model: {method: accuracy}}
    When multiple files map to the same (model, method), the one with the
    most records (or latest name) wins.
    """
    # Accumulate per (model, method): list of records across files
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
        # Keep the file with more records; on ties, the later (sorted) filename wins
        if key not in buckets or len(records) >= len(buckets[key]):
            buckets[key] = records

    data: dict[str, dict[str, float]] = {}
    for (model, method), records in buckets.items():
        accuracy = sum(is_correct(r) for r in records) / len(records)
        data.setdefault(model, {})[method] = accuracy

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _method_sort_key(method: str) -> int:
    """Return sort position based on desired method order.

    Order: Baseline < EmptyCheatsheet < Cumulative < Retrieval < RetrievalSynthesis < StrategicChunk
    More specific keywords are checked first to avoid false matches.
    """
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
    return 6  # unknown methods go last


def plot(dataset_name: str, data: dict[str, dict[str, float]]) -> None:
    all_methods: list[str] = []
    for methods in data.values():
        for m in methods:
            if m not in all_methods:
                all_methods.append(m)
    all_methods.sort(key=_method_sort_key)

    models = list(data.keys())
    n_models = len(models)
    n_methods = len(all_methods)

    x = np.arange(n_methods)
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(8, n_methods * 1.5), 5))

    for i, model in enumerate(models):
        accuracies = [data[model].get(method, float("nan")) for method in all_methods]
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, accuracies, bar_width, label=model)
        for bar, acc in zip(bars, accuracies):
            if not np.isnan(acc):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{acc:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(all_methods, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Accuracy on {dataset_name}")
    ax.legend(title="Model")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()

    safe_name = dataset_name.replace("/", "_")
    out_path = Path("figures") / f"{safe_name}_summary.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark results for a dataset.")
    parser.add_argument("--datasetname", required=True, help="Dataset subdirectory under results/")
    args = parser.parse_args()

    results_dir = Path("results") / args.datasetname
    if not results_dir.is_dir():
        raise SystemExit(f"Directory not found: {results_dir}")

    data = load_results(results_dir)
    if not data:
        raise SystemExit("No results loaded — check that the directory contains .jsonl files.")

    # Print summary table
    all_methods: list[str] = []
    for methods in data.values():
        for m in methods:
            if m not in all_methods:
                all_methods.append(m)
    all_methods.sort(key=_method_sort_key)

    col_w = max(len(m) for m in all_methods) + 2
    header = f"{'Model':<25}" + "".join(f"{m:>{col_w}}" for m in all_methods)
    print(f"\n{args.datasetname}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for model, methods in data.items():
        row = f"{model:<25}" + "".join(
            f"{methods.get(m, float('nan')):>{col_w}.1%}"
            if not np.isnan(methods.get(m, float("nan")))
            else f"{'N/A':>{col_w}}"
            for m in all_methods
        )
        print(row)
    print()

    plot(args.datasetname, data)


if __name__ == "__main__":
    main()
