#!/bin/bash
cd "/Users/jerrygu/Jerry/ML Research/Dynamic Cheatsheet/dynamic-cheatsheet"
python run_benchmark.py --model_name "openai/gpt-4o-2024-11-20" --task "AIME_2020_2024" --approach_name "DynamicCheatsheet_StrategicChunkRetrieval" --generator_prompt_path "prompts/generator_prompt.txt" --cheatshet_prompt_path "prompts/curator_prompt_for_dc_cumulative.txt" --save_directory "jerry_experiments"
