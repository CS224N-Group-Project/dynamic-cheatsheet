#!/usr/bin/env python3
"""
Convert data/DataSIR/DataSIR.json into a HuggingFace Dataset saved at data/DataSIR/.

Fields saved:
  input           - the encoded/obfuscated string (Encoded_data)
  target          - the sensitive information category (Sensitive_data)
  original_data   - the plain-text value before encoding
  encoding_method - the obfuscation technique applied
  data_type       - General | English | Chinese
"""
import json
from datasets import Dataset, Features, Value

TARGET_FEATURES = Features({
    "input":           Value("string"),
    "target":          Value("string"),
    "original_data":   Value("string"),
    "encoding_method": Value("string"),
    "data_type":       Value("string"),
})

SRC = "data/DataSIR/DataSIR.json"
DST = "data/DataSIR"

print(f"Loading {SRC}...")
with open(SRC) as f:
    raw = json.load(f)
print(f"  {len(raw):,} records loaded")

rows = [
    {
        "input":           r["Encoded_data"],
        "target":          r["Sensitive_data"],
        "original_data":   r["Original_data"],
        "encoding_method": r["Encoding_method"],
        "data_type":       r["Data_type"],
    }
    for r in raw
]

ds = Dataset.from_list(rows, features=TARGET_FEATURES)
ds.save_to_disk(DST)
print(f"Saved {len(ds):,} examples to {DST}/")
