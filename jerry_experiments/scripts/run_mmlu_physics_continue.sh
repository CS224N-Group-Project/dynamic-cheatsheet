#!/bin/bash
cd "/Users/jerrygu/Jerry/ML Research/Dynamic Cheatsheet/dynamic-cheatsheet"
python run_benchmark.py --model_name "openai/gpt-4o-2024-11-20" --task "MMLU_Pro_Physics" --approach_name "DynamicCheatsheet_StrategicChunkRetrieval" --generator_prompt_path "prompts/generator_prompt.txt" --cheatshet_prompt_path "prompts/curator_prompt_for_dc_cumulative.txt" --save_directory "jerry_experiments" --max_n_samples 250 --continue_from_last_run_path "jerry_experiments/MMLU_Pro_Physics/openai/gpt-4o-2024-11-20_DynamicCheatsheet_StrategicChunkRetrieval_2026-02-18-16-41_.jsonl"
