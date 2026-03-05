#!/usr/bin/env python3
"""
Download and merge IneqMath train + dev splits into data/IneqMath_all.
Test split is excluded (ground truth not publicly available).
Loads directly from parquet to avoid HF schema bugs.
"""
import json
import pandas as pd
from datasets import Dataset, Features, Value
from huggingface_hub import hf_hub_download

TARGET_FEATURES = Features({
    "input":    Value("string"),
    "target":   Value("string"),
    "data_id":  Value("string"),
    "type":     Value("string"),
    "choices":  Value("string"),
    "solution": Value("string"),
})

def load_split(filename: str) -> list[dict]:
    path = hf_hub_download(repo_id="AI4Math/IneqMath", filename=filename, repo_type="dataset")
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else list(data.values())

def normalize(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {}
        for col in TARGET_FEATURES:
            val = r.get(col, "")
            if val is None:
                val = ""
            if not isinstance(val, str):
                val = json.dumps(val)
            row[col] = val
        rows.append(row)
    return pd.DataFrame(rows)

print("Downloading train split...")
train_records = load_split("json/train.json")
print(f"  train: {len(train_records)} records")

print("Downloading dev split...")
dev_records = load_split("json/dev.json")
print(f"  dev:   {len(dev_records)} records")

df = pd.concat([normalize(train_records), normalize(dev_records)], ignore_index=True)
print(f"merged: {len(df)} rows")

merged = Dataset.from_pandas(df, features=TARGET_FEATURES)
merged.save_to_disk("data/IneqMath_all")
print("Saved to data/IneqMath_all")
