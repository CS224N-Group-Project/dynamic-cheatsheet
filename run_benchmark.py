import argparse
from datetime import datetime
import json
import os
import sys

PREDEFINED_PROMPTS = {
    "GameOf24": f"Let's play a game called 24. You'll be given four integers, and your objective is to use each number only once, combined with any of the four arithmetic operations (addition, subtraction, multiplication, and division) and parentheses, to achieve a total of 24. For example, if the input is 4, 7, 8, and 8, the output could be (7 - (8 / 8)) * 4 = 24. Please present a single expression that evaluates to 24.",
}

CLI_ALIASES = {
    "--approach": "--approach_name",
    "--cheatshet_prompt_path": "--cheatsheet_prompt_path",
}

ARGUMENT_FIELDS = [
    "task",
    "approach_name",
    "model_name",
    "generator_prompt_path",
    "cheatsheet_prompt_path",
    "max_tokens",
    "temperature",
    "max_num_rounds",
    "execute_python_code",
    "initialize_cheatsheet_path",
    "retrieve_top_k",
    "save_directory",
    "additional_flag_for_save_path",
    "max_n_samples",
    "no_shuffle",
    "prob",
    "noise_n",
    "pregenerated_memory_path",
    "memory_generator_prompt_path",
]


def str_to_bool(value: str) -> bool:
    """
    Parse common boolean text values.
    """
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for benchmark runs.
    """
    parser = argparse.ArgumentParser(description="Run Dynamic Cheatsheet benchmarks.")
    parser.add_argument("--task", default="GameOf24")
    parser.add_argument("--approach_name", default="DynamicCheatsheet_Cumulative")
    parser.add_argument("--model_name", default="openai/gpt-4o-mini")
    parser.add_argument("--generator_prompt_path", default="prompts/generator_prompt.txt")
    parser.add_argument("--cheatsheet_prompt_path", default=None)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_num_rounds", type=int, default=1)
    parser.add_argument("--execute_python_code", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--initialize_cheatsheet_path", default=None)
    parser.add_argument("--retrieve_top_k", type=int, default=3)
    parser.add_argument("--prob", type=float, default=None,
                        help="If set, use softmax(confidence*similarity) to select entries whose cumulative probability exceeds this threshold (e.g. 0.9) instead of top-k.")
    parser.add_argument("--save_directory", default="experiments")
    parser.add_argument("--additional_flag_for_save_path", default="")
    parser.add_argument("--max_n_samples", type=int, default=-1)
    parser.add_argument("--no_shuffle", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--noise_n", type=int, default=None,
                        help="Number of random noise memory items for IsolatedMemory scenario 2b. Falls back to --retrieve_top_k if not set.")
    parser.add_argument("--pregenerated_memory_path", default=None,
                        help="Path to JSON file for saving/loading pre-generated memory items (IsolatedMemory).")
    parser.add_argument("--memory_generator_prompt_path", default="prompts/isolated_memory_generator_prompt.txt",
                        help="Prompt template for generating per-question memory items (IsolatedMemory).")
    return parser


def args_to_dict(args: argparse.Namespace) -> dict:
    """
    Convert parsed args into a stable, serializable dict.
    """
    return {field: getattr(args, field) for field in ARGUMENT_FIELDS}


def read_file(file_path: str) -> str:
    """
    Read the file and return the content.
    """
    with open(file_path, "r") as file:
        return file.read()

    
def write_jsonl(file_path, data):
    """
    Save the outputs to a file.
    """
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "w") as file:
        for line in data:
            file.write(json.dumps(line) + "\n")


def normalize_cli_args(argv):
    """
    Normalize CLI aliases so docs and legacy flags both work.
    """
    normalized = []
    for arg in argv:
        if "=" in arg and arg.startswith("--"):
            key, value = arg.split("=", 1)
            normalized_key = CLI_ALIASES.get(key, key)
            normalized.append(f"{normalized_key}={value}")
        else:
            normalized.append(CLI_ALIASES.get(arg, arg))
    return normalized


def main(args: argparse.Namespace):
    """
    Main function to run the benchmark.
    """
    try:
        import pandas as pd
        import numpy as np
        from datasets import load_dataset, load_from_disk
        from dotenv import load_dotenv
        from dynamic_cheatsheet.language_model import LanguageModel
        from dynamic_cheatsheet.utils.evaluation import (
            eval_equation_balancer,
            eval_for_exact_matching_with_no_punctuation,
            eval_for_GameOf24,
            eval_for_ineqmath,
            eval_for_multiple_choice,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        raise ModuleNotFoundError(
            f"Missing dependency '{missing}'. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    # Load the environment variables
    load_dotenv("config.env")

    # Read the prompt files
    args.generator_prompt = read_file(args.generator_prompt_path)
    if args.cheatsheet_prompt_path:
        args.cheatsheet_prompt = read_file(args.cheatsheet_prompt_path)
    else:
        args.cheatsheet_prompt = "(empty)"

    if args.approach_name == "IsolatedMemory":
        args.memory_generator_prompt = read_file(args.memory_generator_prompt_path)
    else:
        args.memory_generator_prompt = None

    # Initialize the language model
    model = LanguageModel(
        model_name=args.model_name,
    )

    # Load self-correction template if using a self-correction approach
    if args.approach_name in ["DynamicCheatsheet_SelfCorrection", "DynamicCheatsheet_SelfCorrection_Improved", "self-correction-improved"]:
        self_correction_path = os.path.join(os.path.dirname(__file__), "prompts", "self_correction_prompt.txt")
        model._self_correction_template = read_file(self_correction_path)

    # Add a flag to the save path if the code execution is not allowed
    if not args.execute_python_code:
        args.additional_flag_for_save_path += "_no-code-execution"

    # Load the dataset based on the task name
    if args.task in PREDEFINED_PROMPTS and args.task != "P3_Test":
        dataset = load_dataset("turingmachine/meta-prompting")
        dataset = dataset[args.task]
    elif args.task in ["GPQA_Diamond", "AIME_2020_2024", "AIME_2024", "AIME_2025", "MMLU_Pro_Physics", "MMLU_Pro_Engineering", "MathEquationBalancer", "IneqMath", "IneqMath_test", "IneqMath_dev"]:
        dataset = load_from_disk(f"data/{args.task}")
    else:
        raise ValueError(f"Task {args.task} is not recognized. Please make sure the task name is correct.")
    
    # Resolve noise_n: explicit --noise_n takes priority, else fall back to --retrieve_top_k
    if args.noise_n is None:
        args.noise_n = args.retrieve_top_k

    # Build the deterministic save path (always, up front)
    _retrieval_approaches = {"Dynamic_Retrieval", "DynamicCheatsheet_RetrievalSynthesis", "DynamicCheatsheet_StrategicChunkRetrieval"}
    retrieval_tag = ""
    if args.approach_name in _retrieval_approaches:
        retrieval_tag = f"_prob{args.prob}" if args.prob is not None else f"_topk{args.retrieve_top_k}"
    elif args.approach_name == "IsolatedMemory":
        retrieval_tag = f"_noise{args.noise_n}"
    _safe_model_name = args.model_name.replace("/", "-")
    _flag = f"_{args.additional_flag_for_save_path}" if args.additional_flag_for_save_path else ""
    args.save_path_name = f"{args.save_directory}/{args.task}/{_safe_model_name}_{args.approach_name}{retrieval_tag}{_flag}.jsonl"
    os.makedirs(os.path.dirname(args.save_path_name), exist_ok=True)

    save_param_path = args.save_path_name.replace(".jsonl", "_params.json")

    # Auto-resume if a prior partial run exists
    _is_resume = os.path.exists(args.save_path_name)
    if _is_resume and os.path.exists(save_param_path):
        with open(save_param_path, "r") as file:
            previous_run_params = json.load(file)

        # Validate that key params are consistent with the previous run
        args_keys = ["generator_prompt_path", "cheatsheet_prompt_path", "temperature", "execute_python_code", "task", "model_name", "approach_name", "max_num_rounds"]
        for key in args_keys:
            if key == "cheatsheet_prompt_path":
                if "cheatsheet_prompt_path" in previous_run_params:
                    prev_value = previous_run_params["cheatsheet_prompt_path"]
                elif "cheatshet_prompt_path" in previous_run_params:
                    prev_value = previous_run_params["cheatshet_prompt_path"]
                else:
                    raise ValueError(f"Warning: The provided argument {key} could not be found in the previous run metadata.")
            else:
                if key not in previous_run_params:
                    raise ValueError(f"Warning: The provided argument {key} could not be found in the previous run metadata.")
                prev_value = previous_run_params[key]
            if getattr(args, key) != prev_value:
                raise ValueError(f"Warning: The provided argument {key} is inconsistent with the previous run. The previous run value is {prev_value}.")

    # Save the arguments to a file (run_timestamp records when this run started/resumed)
    with open(save_param_path, "w") as file:
        json.dump({**args_to_dict(args), "run_timestamp": datetime.today().strftime('%Y-%m-%d-%H-%M')}, file, indent=4)

    # Initialize the cheatsheet
    cheatsheet = "(empty)"
    if args.initialize_cheatsheet_path is not None:
        with open(args.initialize_cheatsheet_path, "r") as file:
            cheatsheet = file.read()

    # Initialize the outputs and the generator outputs so far
    outputs = []
    generator_outputs_so_far = []
    if _is_resume:
        # Load the previous run
        with open(args.save_path_name, "r") as file:
            outputs = [json.loads(line) for line in file.readlines()]

        # Load the previous cheatsheet from the last output
        cheatsheet = outputs[-1]["final_cheatsheet"]

        # Re-embed memory store on resume (StrategicChunkRetrieval stores source_input, not embeddings, on disk)
        if args.approach_name == "DynamicCheatsheet_StrategicChunkRetrieval" and cheatsheet not in (None, "(empty)"):
            try:
                _resume_store = json.loads(cheatsheet)
            except Exception:
                _resume_store = []
            if _resume_store:
                print(f"Re-embedding {len(_resume_store)} memory items for resume...")
                for _item in _resume_store:
                    _item["embedding"] = model._embed_text(_item["text"])
                cheatsheet = json.dumps(_resume_store)

        generator_outputs_so_far = [output["final_output"] for output in outputs]

        # Print the details
        print(f"Continuing from the previous run at {args.save_path_name}.")
        print(f"Loaded {len(outputs)} examples from the previous run.")
        print(f"Most recent cheatsheet: {cheatsheet}")
        print("-" * 50)

    # Split the dataset by taking the first n samples
    # dataset = dataset.select(range(args.max_n_samples))

    # Shuffle the dataset if the no_shuffle flag is not set
    if not args.no_shuffle:
        dataset = dataset.shuffle(seed=10)

    # Initialize the questions and the embeddings
    questions = None
    embeddings = None
    if args.approach_name in [
        "Dynamic_Retrieval",
        "DynamicCheatsheet_RetrievalSynthesis",
        "FullHistoryAppending",
        "DynamicCheatsheet_SelfCorrection",
        "DynamicCheatsheet_SelfCorrection_Improved",
        "self-correction-improved",
    ]:
        df = pd.read_csv(f"embeddings/{args.task}.csv")
        questions = df["input"].tolist()
        embeddings = df["embedding"]
        embeddings = embeddings.apply(eval)
        embeddings = np.array(embeddings.tolist()) # (N, 1536)

        # Re-order the embeddings based on the order of the dataset inputs
        dataset_inputs = [example["input"] for example in dataset]  # type: ignore[index]
        indices = [questions.index(input) for input in dataset_inputs]
        embeddings = embeddings[indices]
        questions = dataset_inputs
    else:
        questions = [example["input"] for example in dataset]  # type: ignore[index]

    # IsolatedMemory: pre-generate (or load) one memory item per question
    pregenerated_memory_items = None
    if args.approach_name == "IsolatedMemory":
        from dynamic_cheatsheet.utils.extractor import extract_all_memory_items, extract_cheatsheet

        if args.pregenerated_memory_path:
            memory_path = args.pregenerated_memory_path
        else:
            memory_path = f"{args.save_directory}/{args.task}/{_safe_model_name}_pregenerated_memory.json"

        os.makedirs(os.path.dirname(memory_path), exist_ok=True)

        if os.path.exists(memory_path):
            with open(memory_path, "r") as f:
                pregenerated_memory_items = json.load(f)
            print(f"Loaded {len(pregenerated_memory_items)} pre-generated memory items from {memory_path}.")
        else:
            pregenerated_memory_items = {}

        # Build the full list of formatted questions (same formatting as the eval loop)
        all_formatted_questions = []
        for _qi, _example in enumerate(dataset):
            _orig = _example["input"]
            if args.task in PREDEFINED_PROMPTS:
                _q = f"{PREDEFINED_PROMPTS[args.task]}\n\nQuestion #{_qi+1}:\n{_orig}"
            else:
                _q = f"Question #{_qi+1}:\n{_orig}"
            if args.task in ("AIME_2020_2024", "AIME_2024", "AIME_2025"):
                _q = f"{_q} (Please provide your answer in the form of an integer, e.g., 1234, with no Markdown formatting or additional text; make sure to pay attention to the desired format of the final answer though.)"
            elif args.task == "MathEquationBalancer":
                _q = f"Below is an equation with missing operators. Your task is to fill in the blanks with the correct mathematical operators: +, -, *, or /. Ensure that the equation is correct once the operators are added. The operators should be placed in the sequence they appear from left to right. Include the full equation with the operators filled in. For instance, for the equation 1 ? 2 ? 3 = 6, the correct answer is 1 + 2 + 3 = 6.\n\nEquation: {_q}"
            elif args.task in ("IneqMath", "IneqMath_test", "IneqMath_dev"):
                _ptype = dataset[_qi]["type"]
                if _ptype == "relation":
                    import json as _json
                    _choices_raw = dataset[_qi]["choices"]
                    _choices = _json.loads(_choices_raw) if _choices_raw else []
                    _choices_str = "\n".join(_choices) if _choices else ""
                    _q = f"{_q}\n\nChoices:\n{_choices_str}\n\n(Select the correct relation from the choices above. State your final answer as the choice letter, e.g. (A).)"
                else:
                    _q = f"{_q}\n\n(Provide your final answer as the exact value of the constant, e.g. C = 4.)"
            all_formatted_questions.append(_q)
            if args.max_n_samples > 0 and _qi == args.max_n_samples - 1:
                break

        # Generate missing memory items
        missing = [q for q in all_formatted_questions if q not in pregenerated_memory_items]
        if missing:
            print(f"Generating memory items for {len(missing)} questions...")
            for _mi, _q in enumerate(missing):
                print(f"  Pre-generating memory item {_mi+1}/{len(missing)}...")
                mem_prompt = args.memory_generator_prompt.replace("[[QUESTION]]", _q)
                mem_output = model.generate(
                    history=[{"role": "user", "content": mem_prompt}],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    allow_code_execution=False,
                )
                mem_cheatsheet = extract_cheatsheet(mem_output, old_cheatsheet="(empty)")
                mem_items = extract_all_memory_items(mem_cheatsheet)
                pregenerated_memory_items[_q] = "\n\n".join(mem_items) if mem_items else mem_cheatsheet

            with open(memory_path, "w") as f:
                json.dump(pregenerated_memory_items, f, indent=2)
            print(f"Saved {len(pregenerated_memory_items)} memory items to {memory_path}.")

    start_idx = len(outputs)
    correct_so_far = 0
    total_so_far = 0
    previous_inputs = []

    # Iterate over the dataset
    for idx, example in enumerate(dataset):
        original_input = str(dataset[idx]["input"])
        original_target = str(dataset[idx]["target"])
        orig_input = example["input"]  # type: ignore[index]
        if args.task in PREDEFINED_PROMPTS:
            input = f"{PREDEFINED_PROMPTS[args.task]}\n\nQuestion #{idx+1}:\n{orig_input}"
        else:
            input = f"Question #{idx+1}:\n{orig_input}"

        previous_inputs.append(input)

        if args.task == "AIME_2020_2024" or args.task == "AIME_2024" or args.task == "AIME_2025":
            # Add a specific format to the input for the AIME tasks
            input = f"{input} (Please provide your answer in the form of an integer, e.g., 1234, with no Markdown formatting or additional text; make sure to pay attention to the desired format of the final answer though.)"
        elif args.task == "MathEquationBalancer":
            # Add a specific format to the input for the MathEquationBalancer task
            input = f"Below is an equation with missing operators. Your task is to fill in the blanks with the correct mathematical operators: +, -, *, or /. Ensure that the equation is correct once the operators are added. The operators should be placed in the sequence they appear from left to right. Include the full equation with the operators filled in. For instance, for the equation 1 ? 2 ? 3 = 6, the correct answer is 1 + 2 + 3 = 6.\n\nEquation: {input}"
        elif args.task in ["IneqMath", "IneqMath_test", "IneqMath_dev"]:
            problem_type = dataset[idx]["type"]
            if problem_type == "relation":
                import json as _json
                choices_raw = dataset[idx]["choices"]
                choices = _json.loads(choices_raw) if choices_raw else []
                choices_str = "\n".join(choices) if choices else ""
                input = f"{input}\n\nChoices:\n{choices_str}\n\n(Select the correct relation from the choices above. State your final answer as the choice letter, e.g. (A).)"
            else:
                input = f"{input}\n\n(Provide your final answer as the exact value of the constant, e.g. C = 4.)"

        # Skip the examples that have been already seen in the previous run
        if idx < start_idx:
            continue

        # Print the details
        print(f"### Example {idx+1} ###")
    
        # Generate the output from the language model using the DynamicCheatsheet approach or other approaches
        output_dict = model.advanced_generate(
            approach_name=args.approach_name,
            input_txt=input,
            cheatsheet=cheatsheet,
            generator_template=args.generator_prompt,
            cheatsheet_template=args.cheatsheet_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_num_rounds=args.max_num_rounds,
            allow_code_execution=args.execute_python_code,
            code_execution_flag="EXECUTE CODE!",
            original_input_corpus=questions[:idx+1],
            original_input_embeddings=embeddings[:idx+1] if embeddings is not None and args.approach_name in [
                "Dynamic_Retrieval",
                "DynamicCheatsheet_RetrievalSynthesis",
                "FullHistoryAppending",
                "DynamicCheatsheet_StrategicChunkRetrieval",
            ] else None,  # type: ignore[arg-type]
            generator_outputs_so_far=generator_outputs_so_far,
            retrieve_top_k=args.retrieve_top_k,
            retrieve_prob=args.prob,
            pregenerated_memory_items=pregenerated_memory_items,
            noise_n=args.noise_n,
        )

        generator_outputs_so_far.append(output_dict["final_output"])


        # Pop the embeddings-bearing cheatsheet before saving — it carries forward in memory only
        cheatsheet_with_embeddings = output_dict.pop("final_cheatsheet_with_embeddings", None)

        output_record = {
                "input": input,
                "target": original_target,
                "raw_input": original_input,
                **output_dict,
            }
        # Save extra metadata for IneqMath leaderboard submission
        if args.task in ["IneqMath", "IneqMath_test", "IneqMath_dev"]:
            output_record["data_id"] = dataset[idx]["data_id"]
            output_record["problem_type"] = dataset[idx]["type"]
        outputs.append(output_record)
        cheatsheet = cheatsheet_with_embeddings if cheatsheet_with_embeddings is not None else output_dict["final_cheatsheet"]
        final_answer = str(output_dict["final_answer"])

        ## FOR DEBUGGING PURPOSES
        # import pdb; pdb.set_trace()
        print(f"@ CHEATSHEET:\n{cheatsheet}")
        print('- ' * 50)
        print(f"Input: {input}")        
        print(f"Target: {original_target}")
        print(f"Final answer: {final_answer}")
        print("**" * 50)

        if args.task == "GameOf24":
            result = eval_for_GameOf24(original_input, final_answer)
        elif args.task in ["AIME_2025", "AIME_2024", "AIME_2020_2024"]:
            result = eval_for_exact_matching_with_no_punctuation(final_answer.lower(), original_target.lower())
        elif args.task in ["GPQA_Diamond", "MMLU_Pro_Engineering", "MMLU_Pro_Physics"]:
            result = eval_for_multiple_choice(input, final_answer, original_target)
        elif args.task == "MathEquationBalancer":
            result = eval_equation_balancer(original_input, final_answer, original_target)
        elif args.task in ["IneqMath", "IneqMath_test", "IneqMath_dev"]:
            problem_type = dataset[idx]["type"]
            choices_json = dataset[idx]["choices"]
            result = eval_for_ineqmath(problem_type, final_answer, original_target, choices_json, args.model_name)
        else:
            raise ValueError(f"Task {args.task} not supported.")
        
        if result:
            correct_so_far += 1
        total_so_far += 1

        print(f"---- Correct so far: {correct_so_far}/{total_so_far}")
        print("###" * 50)

        # Temporarily save the outputs to a file after each example
        write_jsonl(args.save_path_name, outputs)

        if args.max_n_samples > 0 and idx == args.max_n_samples - 1:
            break
        
    # Save the entire outputs to a file
    write_jsonl(args.save_path_name, outputs)

        
if __name__ == "__main__":
    parser = build_argument_parser()
    normalized_argv = normalize_cli_args(sys.argv[1:])
    args = parser.parse_args(normalized_argv)
    main(args)