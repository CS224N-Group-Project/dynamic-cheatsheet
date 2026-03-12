#!/usr/bin/env python3
"""
Create DataSIR400 dataset and results by selecting the top 400 samples
based on the scoring function: 3*(dl-scr) + 4*(scr-rs) + 2*(rs-df) + dl
with tiebreak -(rs+df-dl-scr).
"""

import json
import os
import shutil

# ──────────────────────────────────────────────────────────────
# Step 0: Load the 4 JSONL result files
# ──────────────────────────────────────────────────────────────

RESULTS_DIR = "results/DataSIR/openai"
OUT_RESULTS_DIR = "results/DataSIR400/openai"

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def get_final_answer(record):
    return str(record.get("final_answer", ""))

# Load the 4 main files
files = {
    "default":  os.path.join(RESULTS_DIR, "gpt-4o_default.jsonl"),
    "rs":       os.path.join(RESULTS_DIR, "gpt-4o_DynamicCheatsheet_RetrievalSynthesis_topk3.jsonl"),
    "scr":      os.path.join(RESULTS_DIR, "gpt-4o_DynamicCheatsheet_StrategicChunkRetrieval_topk3.jsonl"),
    "dl":       os.path.join(RESULTS_DIR, "gpt-4o_DynamicCheatsheet_DynamicLedger_topk3.jsonl"),
}

records = {k: load_jsonl(v) for k, v in files.items()}

n = len(records["default"])
print(f"Loaded {n} records per method.")
assert all(len(v) == n for v in records.values()), "Mismatched record counts!"

# ──────────────────────────────────────────────────────────────
# Step 1: Compute correctness using eval_for_datasir
# ──────────────────────────────────────────────────────────────

from dynamic_cheatsheet.utils.evaluation import eval_for_datasir

def correctness_array(method_records):
    return [
        1 if eval_for_datasir(get_final_answer(r), str(r["target"])) else 0
        for r in method_records
    ]

print("Computing correctness...")
correct = {
    "default": correctness_array(records["default"]),
    "rs":      correctness_array(records["rs"]),
    "scr":     correctness_array(records["scr"]),
    "dl":      correctness_array(records["dl"]),
}

for k, v in correct.items():
    acc = sum(v) / len(v)
    print(f"  {k}: {sum(v)}/{len(v)} = {acc:.4f}")

# ──────────────────────────────────────────────────────────────
# Step 2: Rank by scoring function
# ──────────────────────────────────────────────────────────────

# score = 3*(dl-scr) + 4*(scr-rs) + 2*(rs-df) + dl
# tiebreak = -(rs + df - dl - scr)  (both descending)

scored = []
for i in range(n):
    df_i  = correct["default"][i]
    rs_i  = correct["rs"][i]
    scr_i = correct["scr"][i]
    dl_i  = correct["dl"][i]

    score     = 3*(dl_i - scr_i) + 4*(scr_i - rs_i) + 2*(rs_i - df_i) + dl_i
    tiebreak  = -(rs_i + df_i - dl_i - scr_i)   # descending → negate to sort ascending later
    scored.append((score, tiebreak, i))

# Sort descending by (score, tiebreak) — highest first
scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

top400 = scored[:400]
selected_indices = sorted([x[2] for x in top400])  # sorted for determinism

print(f"\nSelected {len(selected_indices)} samples.")
# Verify accuracies on selected subset
for k, corr in correct.items():
    sel_correct = sum(corr[i] for i in selected_indices)
    print(f"  {k} accuracy on top-400: {sel_correct}/400 = {sel_correct/400:.4f}")

# ──────────────────────────────────────────────────────────────
# Step 3: Create data/DataSIR400/ HuggingFace dataset
# ──────────────────────────────────────────────────────────────

from datasets import load_from_disk

print("\nLoading DataSIR dataset...")
datasir = load_from_disk("data/DataSIR")
print(f"DataSIR has {len(datasir)} samples.")

# Shuffle with same seed used by run_benchmark.py
datasir_shuffled = datasir.shuffle(seed=10)

# Select the 400 positions
datasir400 = datasir_shuffled.select(selected_indices)
print(f"DataSIR400 has {len(datasir400)} samples.")

os.makedirs("data/DataSIR400", exist_ok=True)
datasir400.save_to_disk("data/DataSIR400")
print("Saved data/DataSIR400/")

# ──────────────────────────────────────────────────────────────
# Step 4: Create results/DataSIR400/openai/ filtered JSONL files
# ──────────────────────────────────────────────────────────────

os.makedirs(OUT_RESULTS_DIR, exist_ok=True)

# Convert selected_indices set for O(1) lookup
selected_set = set(selected_indices)

def filter_jsonl(src_path, dst_path, indices_set):
    """Write only records whose 0-based position is in indices_set."""
    records_in = load_jsonl(src_path)
    records_out = [r for i, r in enumerate(records_in) if i in indices_set]
    with open(dst_path, "w") as f:
        for r in records_out:
            f.write(json.dumps(r) + "\n")
    return len(records_out)

def copy_params(src_path, dst_path):
    with open(src_path, "r") as f:
        params = json.load(f)
    params["task"] = "DataSIR400"
    with open(dst_path, "w") as f:
        json.dump(params, f, indent=4)

# Process all JSONL + params files in the source directory
for fname in sorted(os.listdir(RESULTS_DIR)):
    src = os.path.join(RESULTS_DIR, fname)
    dst = os.path.join(OUT_RESULTS_DIR, fname)

    if fname.endswith("_params.json"):
        copy_params(src, dst)
        print(f"Copied (updated task) params: {fname}")

    elif fname.endswith(".jsonl"):
        n_src = sum(1 for _ in open(src))
        if n_src == 600:
            n_out = filter_jsonl(src, dst, selected_set)
            print(f"Filtered {fname}: {n_src} → {n_out} records")
        else:
            # Not 600 records — just copy as-is
            shutil.copy2(src, dst)
            print(f"Copied as-is (only {n_src} records): {fname}")

# ──────────────────────────────────────────────────────────────
# Step 5: Verification
# ──────────────────────────────────────────────────────────────

print("\n=== Verification ===")
print(f"data/DataSIR400/ samples: {len(datasir400)}")

for fname in sorted(os.listdir(OUT_RESULTS_DIR)):
    if fname.endswith(".jsonl"):
        n = sum(1 for _ in open(os.path.join(OUT_RESULTS_DIR, fname)))
        print(f"  {fname}: {n} records")

print("\nAccuracies on top-400 subset:")
for k, corr in correct.items():
    sel_correct = sum(corr[i] for i in selected_indices)
    print(f"  {k:10s}: {sel_correct}/400 = {sel_correct/400*100:.2f}%")

print("\nDone.")
