# Dynamic Cheatsheet - Project Context

## What This Project Is
A CS224n (Stanford) research project building on the Dynamic Cheatsheet paper. The framework gives black-box LLMs persistent, evolving memory during inference via a two-LLM-role architecture.

## Architecture Overview

### Two Roles (same LLM, different prompts)
- **Generator**: Solves problems using the cheatsheet as reference. Can write and execute Python code.
- **Curator**: After each problem, rewrites the entire cheatsheet from scratch based on the previous cheatsheet + the question + the generator's answer.

### Core Loop (DC-Cumulative)
```
cheatsheet = "(empty)"
for each problem in dataset:
    generator_answer = Generator(question, cheatsheet)
    new_cheatsheet = Curator(question, generator_answer, cheatsheet)
    cheatsheet = new_cheatsheet
```

## Key Files

### Core Code
- `dynamic_cheatsheet/language_model.py` — The entire engine. Contains:
  - `generate()` (line 66): Low-level LLM call + recursive code execution loop. Calls the LLM, if output contains "EXECUTE CODE!" it runs the Python code and recursively calls itself (up to 3 rounds).
  - `advanced_generate()` (line 157): High-level orchestrator that branches on `approach_name` and manages the generator→curator pipeline.
  - Line 164: `max_tokens` default changed from 2048 to 4096 (curator gets 2x this).

### Utilities
- `dynamic_cheatsheet/utils/extractor.py` — Parses `<answer>` tags and `<cheatsheet>` tags from LLM output. If `<cheatsheet>` tags missing, falls back to old cheatsheet.
- `dynamic_cheatsheet/utils/execute_code.py` — Extracts ```python blocks, writes to temp file, runs `python3` with 3s timeout.
- `dynamic_cheatsheet/utils/evaluation.py` — Task-specific eval functions (GameOf24, AIME, GPQA, MMLU-Pro, MathEquationBalancer).

### Prompts
- `prompts/generator_prompt.txt` — Generator template. Placeholders: `[[CHEATSHEET]]`, `[[QUESTION]]`.
- `prompts/curator_prompt_for_dc_cumulative.txt` — Curator template for DC-Cumulative. Placeholders: `[[PREVIOUS_CHEATSHEET]]`, `[[QUESTION]]`, `[[MODEL_ANSWER]]`.
- `prompts/curator_prompt_for_dc_retrieval_synthesis.txt` — Curator template for DC-RetrievalSynthesis. Placeholders: `[[PREVIOUS_INPUT_OUTPUT_PAIRS]]`, `[[NEXT_INPUT]]`, `[[PREVIOUS_CHEATSHEET]]`.

### Entry Points
- `run_benchmark.py` — CLI entry point for full benchmark runs. Note: has typo `--cheatshet_prompt_path`.
- `ExampleUsage.ipynb` — Interactive demo (3 rounds of DC-Cumulative on Game of 24).
- `EvaluatingResults.ipynb` — Reads all JSONL result files and prints accuracy per file.

### Data & Results
- `data/` — Local HuggingFace datasets (AIME, GPQA, MMLU-Pro, MathEquationBalancer).
- `embeddings/*.csv` — Precomputed OpenAI embeddings (1536-dim) for retrieval approaches.
- `results/` — JSONL result files organized by task, named `{model}_{approach}.jsonl`.
- `config.env` — API keys (OPENAI_API_KEY, not committed).

## The 5 Approaches

1. **`default`** — Baseline. No cheatsheet, just generator.
2. **`DynamicCheatsheet_Cumulative`** — Generator solves with cheatsheet, curator rewrites entire cheatsheet after each problem. Knowledge accumulates sequentially.
3. **`FullHistoryAppending`** — Dumps all previous Q&A pairs verbatim into prompt. No curator. Scales poorly.
4. **`Dynamic_Retrieval`** — Cosine similarity finds top-k similar past problems, passes raw Q&A pairs to generator. No curator.
5. **`DynamicCheatsheet_RetrievalSynthesis`** — Same retrieval, but curator synthesizes retrieved pairs + old cheatsheet into a tailored cheatsheet BEFORE generator runs.

## Cheatsheet Format
The cheatsheet is a plain text string. The `<memory_item>`, `<description>`, `<example>` tags are LLM formatting conventions from the curator prompt — the code never parses them. Only the outer `<cheatsheet>...</cheatsheet>` tags are parsed by `extract_cheatsheet()`. The curator decides what to add, remove, merge, or update — there's no programmatic enforcement.

## Important Details
- `max_tokens` at line 164 of language_model.py: Generator gets `max_tokens`, Curator gets `2*max_tokens`. Was 2048, changed to 4096. If curator doesn't get enough tokens, it can't output `<cheatsheet>` tags and the cheatsheet stays unchanged.
- The model used for benchmark results: `anthropic/claude-3-5-sonnet-20241022` and `openai/gpt-4o-2024-11-20`. The Claude snapshot was retired Feb 19, 2026.
- Current working model: `openai/gpt-4o-2024-11-20`.
- After changing code in language_model.py, you must restart the Jupyter kernel for changes to take effect.

## Planned New Approach
`DynamicCheatsheet_StrategicChunkRetrieval` (stub at lines 433-440):
- Step 1: Retrieve top-k relevant memory items from cheatsheet during generation
- Step 2: Give retrieved strategies to generator LLM
- Step 3: After generation, decide whether to create new memory items, fix existing ones, or do nothing. Make separate curator calls per change.

## Running the Code

### Example notebook
```bash
cd dynamic-cheatsheet
jupyter notebook ExampleUsage.ipynb
```

### Full benchmark
```bash
python run_benchmark.py \
  --model_name "openai/gpt-4o-2024-11-20" \
  --task "AIME_2024" \
  --approach_name "DynamicCheatsheet_Cumulative" \
  --generator_prompt_path "prompts/generator_prompt.txt" \
  --cheatshet_prompt_path "prompts/curator_prompt_for_dc_cumulative.txt"
```

### Evaluate results
```bash
jupyter notebook EvaluatingResults.ipynb
```
Then Run All. Prints accuracy for every result file found.
