#!/usr/bin/env python3
"""
build_embeddings.py --task <name>
Pre-compute text-embedding-3-small embeddings for a dataset and save to
embeddings/<task>.csv, matching the format expected by run_benchmark.py.

Shuffle and sample options mirror run_benchmark.py defaults (seed=10) so
the embedded subset exactly matches what the benchmark will iterate over.
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from dotenv import load_dotenv
from litellm import embedding as litellm_embedding
from litellm.exceptions import RateLimitError


def embed_text(text: str, retries: int = 8) -> list[float]:
    for attempt in range(retries):
        try:
            response = litellm_embedding(
                model="openai/text-embedding-3-small",
                input=[text],
            )
            return response.data[0]["embedding"]
        except RateLimitError as e:
            wait = min(2 ** attempt * 5, 120)
            print(f"Rate limit hit, retrying in {wait}s... ({e})")
            time.sleep(wait)
    raise RateLimitError("Max retries exceeded due to rate limiting.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embeddings CSV for a dataset.")
    parser.add_argument("--task", required=True, help="Dataset name under data/")
    parser.add_argument("--max_n_samples", type=int, default=-1,
                        help="Embed only the first N samples after shuffling "
                             "(use the same value as --max_n_samples in run_benchmark.py). "
                             "-1 means embed all.")
    parser.add_argument("--seed", type=int, default=10,
                        help="Shuffle seed (must match run_benchmark.py, default 10).")
    parser.add_argument("--no_shuffle", action="store_true",
                        help="Skip shuffling (must match run_benchmark.py --no_shuffle).")
    args = parser.parse_args()

    load_dotenv("config.env")

    dataset = load_from_disk(f"data/{args.task}")

    if not args.no_shuffle:
        dataset = dataset.shuffle(seed=args.seed)

    if args.max_n_samples > 0:
        dataset = dataset.select(range(args.max_n_samples))
        print(f"Using {len(dataset)} samples (shuffled with seed={args.seed}).")

    out_path = Path("embeddings") / f"{args.task}.csv"
    out_path.parent.mkdir(exist_ok=True)

    # Resume from existing file if present
    done_inputs: set[str] = set()
    rows: list[dict] = []
    if out_path.exists():
        existing = pd.read_csv(out_path)
        for _, row in existing.iterrows():
            done_inputs.add(row["input"])
            rows.append({"input": row["input"], "embedding": row["embedding"]})
        print(f"Resuming: {len(rows)} already embedded.")

    total = len(dataset)
    for i, example in enumerate(dataset):
        text = example["input"]
        if text in done_inputs:
            continue
        print(f"[{i+1}/{total}] Embedding...")
        emb = embed_text(text)
        rows.append({"input": text, "embedding": json.dumps(emb)})
        done_inputs.add(text)

        # Save incrementally every 10 items
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {len(rows)} embeddings to {out_path}")


if __name__ == "__main__":
    main()
