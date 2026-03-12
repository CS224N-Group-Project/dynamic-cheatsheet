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


def _build_datasir_from_gz(
    gz_path: str = "data/DataSIR/DataSIR.json.gz",
    dst: str = "data/DataSIR",
) -> None:
    """
    Build the HuggingFace Dataset from the compressed source file.
    Called automatically on first use when the arrow files are absent.
    """
    import gzip, json as _json
    from datasets import Dataset, Features, Value

    print(f"[DataSIR] Building dataset from {gz_path} (first-time setup, ~30 s)...")
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        raw = _json.load(f)
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
    features = Features({
        "input":           Value("string"),
        "target":          Value("string"),
        "original_data":   Value("string"),
        "encoding_method": Value("string"),
        "data_type":       Value("string"),
    })
    Dataset.from_list(rows, features=features).save_to_disk(dst)
    print(f"[DataSIR] Saved {len(rows):,} examples to {dst}/")


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
            eval_for_datasir,
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
    _DEFAULT_CURATOR_PROMPTS = {
        "DynamicCheatsheet_JSON_Memory": "prompts/curator_prompt_json_memory.txt",
    }
    if args.cheatsheet_prompt_path:
        args.cheatsheet_prompt = read_file(args.cheatsheet_prompt_path)
    elif args.approach_name in _DEFAULT_CURATOR_PROMPTS:
        args.cheatsheet_prompt = read_file(_DEFAULT_CURATOR_PROMPTS[args.approach_name])
    else:
        args.cheatsheet_prompt = "(empty)"

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
    elif args.task in ["GPQA_Diamond", "AIME_2020_2024", "AIME_2024", "AIME_2025", "MMLU_Pro_Physics", "MMLU_Pro_Engineering", "MathEquationBalancer", "IneqMath", "IneqMath_test", "IneqMath_dev", "IneqMath_all", "DataSIR"]:
        if args.task == "DataSIR" and not os.path.exists("data/DataSIR/data-00000-of-00001.arrow"):
            _build_datasir_from_gz()
        dataset = load_from_disk(f"data/{args.task}")
    else:
        raise ValueError(f"Task {args.task} is not recognized. Please make sure the task name is correct.")
    
    # Build the deterministic save path (always, up front)
    _retrieval_approaches = {"Dynamic_Retrieval", "DynamicCheatsheet_RetrievalSynthesis", "DynamicCheatsheet_StrategicChunkRetrieval", "DynamicCheatsheet_JSON_Memory"}
    retrieval_tag = ""
    if args.approach_name in _retrieval_approaches:
        retrieval_tag = f"_prob{args.prob}" if args.prob is not None else f"_topk{args.retrieve_top_k}"
    _model_parts = args.model_name.split("/")
    if len(_model_parts) == 2:
        _provider, _model_slug = _model_parts
        _save_dir = f"{args.save_directory}/{args.task}/{_provider}"
        _safe_model_name = _model_slug
    else:
        _save_dir = f"{args.save_directory}/{args.task}"
        _safe_model_name = args.model_name
    _flag = f"_{args.additional_flag_for_save_path}" if args.additional_flag_for_save_path else ""
    args.save_path_name = f"{_save_dir}/{_safe_model_name}_{args.approach_name}{retrieval_tag}{_flag}.jsonl"
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

        # Re-embed memory store on resume for JSON Memory (two embeddings per item: strategy + problem)
        elif args.approach_name == "DynamicCheatsheet_JSON_Memory" and cheatsheet not in (None, "(empty)"):
            try:
                _resume_store = json.loads(cheatsheet)
            except Exception:
                _resume_store = []
            if _resume_store:
                print(f"Re-embedding {len(_resume_store)} memory items for resume (JSON Memory)...")
                # Batch all texts in one API call: [strategy_0, problem_0, strategy_1, problem_1, ...]
                _all_texts = []
                for _item in _resume_store:
                    _all_texts.append(_item["strategy"])
                    _all_texts.append(_item["example_problem"])
                _all_embeddings = model._embed_batch(_all_texts)
                for i, _item in enumerate(_resume_store):
                    _item["strategy_embedding"] = _all_embeddings[2 * i]
                    _item["problem_embedding"]  = _all_embeddings[2 * i + 1]
                cheatsheet = json.dumps(_resume_store)

        generator_outputs_so_far = [output["final_output"] for output in outputs]

        # Print the details
        print(f"Continuing from the previous run at {args.save_path_name}.")
        print(f"Loaded {len(outputs)} examples from the previous run.")
        print(f"Most recent cheatsheet: {cheatsheet}")
        print("-" * 50)

    # Shuffle the dataset if the no_shuffle flag is not set
    if not args.no_shuffle:
        dataset = dataset.shuffle(seed=10)

    # Truncate to max_n_samples before building the embeddings index so that
    # the lookup only needs to cover the samples actually used in this run.
    if args.max_n_samples > 0:
        dataset = dataset.select(range(args.max_n_samples))

    # Filter out examples with empty input fields (e.g. malformed entries in IneqMath_all)
    dataset = dataset.filter(lambda x: x["input"] is not None and str(x["input"]).strip() != "")

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
        elif args.task in ["IneqMath", "IneqMath_test", "IneqMath_dev", "IneqMath_all"]:
            problem_type = dataset[idx]["type"]
            if problem_type == "relation":
                import json as _json
                choices_raw = dataset[idx]["choices"]
                choices = _json.loads(choices_raw) if choices_raw else []
                choices_str = "\n".join(choices) if choices else ""
                input = f"{input}\n\nChoices:\n{choices_str}\n\n(Select the correct relation from the choices above. State your final answer as the choice letter, e.g. (A).)"
            else:
                input = f"{input}\n\n(Provide your final answer as the exact value of the constant, e.g. C = 4.)"
        elif args.task == "DataSIR":
            from dynamic_cheatsheet.utils.evaluation import DATASIR_CATEGORIES
            categories_str = ", ".join(DATASIR_CATEGORIES)
            input = (
                f"The following data has been encoded or obfuscated to conceal sensitive personal information. "
                f"Identify which type of sensitive information it contains.\n\n"
                f"Encoded data: {orig_input}\n\n"
                f"Choose exactly one category from this list: {categories_str}.\n"
                f"State your final answer as the category name only, e.g. 'Email'."
            )

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
        if args.task in ["IneqMath", "IneqMath_test", "IneqMath_dev", "IneqMath_all"]:
            output_record["data_id"] = dataset[idx]["data_id"]
            output_record["problem_type"] = dataset[idx]["type"]
        outputs.append(output_record)
        cheatsheet = cheatsheet_with_embeddings if cheatsheet_with_embeddings is not None else output_dict["final_cheatsheet"]
        final_answer = str(output_dict["final_answer"])

        ## FOR DEBUGGING PURPOSES
        # import pdb; pdb.set_trace()
        if args.approach_name == "DynamicCheatsheet_JSON_Memory" and output_dict.get("steps"):
            _step = output_dict["steps"][0]
            _ops  = _step.get("operations_applied", [])
            _n_create = sum(1 for o in _ops if o.get("operation") == "create")
            _n_update = sum(1 for o in _ops if o.get("operation") == "update")
            _n_delete = sum(1 for o in _ops if o.get("operation") == "delete")
            print(f"@ JSON MEMORY — Operations: {_n_create} create | {_n_update} update | {_n_delete} delete  "
                  f"(store size: {output_dict.get('memory_store_size', '?')})")
            print(f"@ MEMORY STORE (strategies):\n{output_dict.get('memory_store_text', '(empty)')}")
        else:
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
        elif args.task in ["IneqMath", "IneqMath_test", "IneqMath_dev", "IneqMath_all"]:
            problem_type = dataset[idx]["type"]
            choices_json = dataset[idx]["choices"]
            result = eval_for_ineqmath(problem_type, final_answer, original_target, choices_json, args.model_name)
        elif args.task == "DataSIR":
            result = eval_for_datasir(final_answer, original_target)
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