#!/usr/bin/env python3
"""
build_embeddings.py --task <name>
Pre-compute text-embedding-3-small embeddings for a dataset and save to
embeddings/<task>.csv, matching the format expected by run_benchmark.py.
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
    args = parser.parse_args()

    load_dotenv("config.env")

    dataset = load_from_disk(f"data/{args.task}")

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
