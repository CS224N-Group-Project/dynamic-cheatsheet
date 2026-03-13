# Dynamic Ledger: Test-Time Learning with Strategic Retrievable Memory



## Overview

Our work is an extension of [Dynamic Cheatsheet (DC), Mirac et al](https://arxiv.org/abs/2504.07952). DC gives language models a persistent, evolving memory during inference. Rather than re-discovering solutions from scratch on every query, DC enables models to accumulate and reuse strategies, code snippets, and problem-solving techniques — without modifying the underlying model parameters.

This repository extends the original DC framework with two new memory architectures and retrieval strategies, two new benchmarks, and support for additional models, and boost the performance of DC significantly on benchmarks.

---

## Our Contributions

### 1. Dynamic Ledger — Structured, Individually Addressable Strategy Store

We introduce **DynamicCheatsheet_DynamicLedger** (the Dynamic Ledger), a new cheatsheet variant that replaces the flat text cheatsheet with a structured JSON store of individually addressable strategy entries.

**How it works:**
- The **generator** embeds the current problem and retrieves the top-*k* most relevant strategy entries from the ledger via cosine similarity.
- The **curator** maintains the ledger after each problem by issuing `create`, `update`, or `delete` operations on individual entries — rather than rewriting the entire cheatsheet.

**Why it's better than DC-Cumulative:**
- Retrieval is precise: only relevant strategies are injected into the context, keeping prompts short.
- Updates are surgical: a single bad strategy can be corrected without disrupting the rest of the store.
- The store scales gracefully — performance does not degrade as the number of accumulated problems grows.

**Run Dynamic Ledger on IneqMath:**
```bash
python3 run_benchmark.py \
  --task IneqMath_all \
  --approach_name DynamicCheatsheet_DynamicLedger \
  --model_name openai/gpt-4o \
  --generator_prompt_path prompts/generator_prompt_dynamic_ledger.txt \
  --cheatsheet_prompt_path prompts/curator_prompt_dynamic_ledger.txt \
  --retrieve_top_k 3 \
  --max_n_samples 600
```

---

### 2. Strategic Chunk Retrieval — Focus-Then-Refine Memory Updates

We introduce **DynamicCheatsheet_StrategicChunkRetrieval**, a retrieval approach that narrows both generation and curation to only the most relevant subset of the cheatsheet.

**How it works:**
- The **generator** embeds the current problem and retrieves the top-*k* most relevant `<memory_item>` chunks from the store via cosine similarity. Only those chunks are shown in the prompt.
- The **curator** receives only the retrieved chunks (not the full cheatsheet) and refines them based on the current problem and model answer. Refined items replace the originals; all other entries remain untouched.
- A **usage counter** per chunk tracks how often each strategy is successfully applied, allowing high-signal strategies to be prioritized over time.

**Why it's better than DC-Cumulative:**
- Context stays bounded: the generator and curator see only relevant chunks, not an ever-growing flat document.
- Targeted updates: only the retrieved chunks are rewritten, leaving the rest of the store stable.
- Usage-aware prioritization: frequently applied strategies accumulate higher counts, surfacing the most reliable patterns.

**Run Strategic Chunk Retrieval on IneqMath:**
```bash
python3 run_benchmark.py \
  --task IneqMath_all \
  --approach_name DynamicCheatsheet_StrategicChunkRetrieval \
  --model_name openai/gpt-4o \
  --cheatsheet_prompt_path prompts/curator_prompt_for_strategic_chunk_retrieval.txt \
  --retrieve_top_k 3 \
  --max_n_samples 600
```

---

### 3. IneqMath Benchmark

We add support for **[IneqMath](https://huggingface.co/datasets/AI4Math/IneqMath)**, a dataset of competition-style inequality problems in two formats:

| Split | Type | Size |
|---|---|---|
| `IneqMath` | dev only (original) | 100 |
| `IneqMath_all` | train + dev merged | 1,352 |

Problem types:
- **bound** — find the largest/smallest constant satisfying an inequality
- **relation** — select the correct inequality direction from multiple-choice options

The `IneqMath_all` dataset is built by `prepare_ineqmath_all.py`, which downloads train and dev splits from HuggingFace, maps fields correctly (`problem`→`input`, `answer`→`target`), and filters out any unlabeled records.

```bash
# Rebuild the dataset from source
python3 prepare_ineqmath_all.py
```

---

### 4. DataSIR Benchmark

We add support for **DataSIR**, a sensitive information recognition dataset. The raw data is large (~460 MB JSON), so we ship a compressed version:

- `data/DataSIR.json.gz` (43 MB) is included in the repo.
- On the first run with `--task DataSIR`, `run_benchmark.py` automatically decompresses and builds the Arrow dataset — no separate prepare step needed.

```bash
python3 run_benchmark.py \
  --task DataSIR \
  --approach_name default \
  --model_name openai/gpt-4o \
  --max_n_samples 600
```

---

## All Supported Approaches

| Approach | Description |
|---|---|
| `default` | No cheatsheet; single-pass generation |
| `DynamicCheatsheet_Cumulative` | Append-only flat text cheatsheet (original DC) |
| `DynamicCheatsheet_RetrievalSynthesis` | Retrieve past examples, synthesize a query-specific cheatsheet |
| `Dynamic_Retrieval` | Retrieve top-*k* chunks, no curation step |
| `FullHistoryAppending` | Full conversation history appended as context |
| `DynamicCheatsheet_StrategicChunkRetrieval` | **[NEW]** Retrieve top-*k* strategy chunks; curator refines only those chunks |
| `DynamicCheatsheet_DynamicLedger` | **[NEW]** Dynamic Ledger — structured JSON store with per-entry CRUD updates |



---

## Supported Models

```
openai/gpt-4o, openai/gpt-4o-mini, openai/gpt-3.5-turbo
openai/gpt-5-2025-08-07
openai/o1, openai/o3-mini
anthropic/claude-3-5-sonnet-latest, anthropic/claude-3-7-sonnet-latest
anthropic/claude-3-5-haiku-latest
xai/grok-3, xai/grok-3-mini, xai/grok-4-fast-non-reasoning
together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo
together_ai/deepseek-ai/DeepSeek-R1, together_ai/Qwen/QwQ-32B
gemini/gemini-2.0-flash
```

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
cp config.env.example config.env  # add your API keys
```

### Basic API Usage

```python
from dynamic_cheatsheet.language_model import LanguageModel

model = LanguageModel(model_name="openai/gpt-4o")

# Dynamic Ledger
with open("prompts/generator_prompt_dynamic_ledger.txt") as f:
    generator_prompt = f.read()
with open("prompts/curator_prompt_dynamic_ledger.txt") as f:
    curator_prompt = f.read()

results = model.advanced_generate(
    approach_name="DynamicCheatsheet_DynamicLedger",  # Dynamic Ledger
    input_txt="<your question>",
    cheatsheet="(empty)",
    generator_template=generator_prompt,
    cheatsheet_template=curator_prompt,
    retrieve_top_k=3,
)

print(results["final_answer"])     # extracted answer
print(results["final_cheatsheet"]) # updated ledger (JSON)
```

### Running Benchmarks

```bash
# Baseline (no cheatsheet)
python3 run_benchmark.py --task IneqMath_all --approach_name default \
    --model_name openai/gpt-4o --max_n_samples 600

# Dynamic Ledger (our method)
python3 run_benchmark.py --task IneqMath_all \
    --approach_name DynamicCheatsheet_DynamicLedger \
    --model_name openai/gpt-4o \
    --generator_prompt_path prompts/generator_prompt_dynamic_ledger.txt \
    --cheatsheet_prompt_path prompts/curator_prompt_dynamic_ledger.txt \
    --retrieve_top_k 3 --max_n_samples 600

# Strategic Chunk Retrieval
python3 run_benchmark.py --task IneqMath_all \
    --approach_name DynamicCheatsheet_StrategicChunkRetrieval \
    --model_name openai/gpt-4o \
    --cheatsheet_prompt_path prompts/curator_prompt_for_strategic_chunk_retrieval.txt \
    --retrieve_top_k 3 --max_n_samples 600
```

Results are saved to `experiments/<task>/<provider>/<model>_<approach>_topk<k>.jsonl`.

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--task` | `GameOf24` | Benchmark task: `IneqMath_all`, `IneqMath`, `DataSIR`, `AIME_2025`, `GPQA_Diamond`, etc. |
| `--approach_name` | `DynamicCheatsheet_Cumulative` | DC variant (see table above) |
| `--model_name` | `openai/gpt-4o-mini` | LLM to use (see Supported Models) |
| `--generator_prompt_path` | `prompts/generator_prompt.txt` | Path to the generator system prompt |
| `--cheatsheet_prompt_path` | `None` | Path to the curator/cheatsheet prompt (required for most DC variants) |
| `--retrieve_top_k` | `3` | Number of strategy chunks to retrieve (retrieval approaches only) |
| `--prob` | `None` | Probability threshold for retrieval (alternative to `--retrieve_top_k`) |
| `--max_tokens` | `2048` | Maximum tokens for generator output |
| `--temperature` | `0.0` | Sampling temperature |
| `--max_num_rounds` | `1` | Number of generation rounds per problem |
| `--execute_python_code` | `True` | Whether to execute Python code blocks in model output |
| `--initialize_cheatsheet_path` | `None` | Path to a pre-built cheatsheet to start from instead of an empty one |
| `--max_n_samples` | `-1` | Cap on examples to process; `-1` for the full dataset |
| `--no_shuffle` | `False` | Disable dataset shuffling (default: shuffle with seed 10) |
| `--save_directory` | `experiments` | Root directory for saving results |
| `--additional_flag_for_save_path` | `""` | Extra suffix appended to the output filename |

---

