#!/bin/bash
cd "/Users/jerrygu/Jerry/ML Research/Dynamic Cheatsheet/dynamic-cheatsheet"
CONTINUE_FILE=$(ls -t jerry_experiments/AIME_2024/openai/gpt-4o-2024-11-20_DynamicCheatsheet_StrategicChunkRetrieval_*.jsonl 2>/dev/null | grep -v "_continued" | grep -v "_params" | head -1)
echo "Continuing from: $CONTINUE_FILE"
python run_benchmark.py --model_name "openai/gpt-4o-2024-11-20" --task "AIME_2024" --approach_name "DynamicCheatsheet_StrategicChunkRetrieval" --generator_prompt_path "prompts/generator_prompt.txt" --cheatshet_prompt_path "prompts/curator_prompt_for_dc_cumulative.txt" --save_directory "jerry_experiments" --continue_from_last_run_path "$CONTINUE_FILE"
