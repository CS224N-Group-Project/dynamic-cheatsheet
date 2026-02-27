import numpy as np
import tiktoken
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from .utils.execute_code import extract_and_run_python_code
from .utils.extractor import extract_answer, extract_cheatsheet
from .utils.confidence import ConfidenceScorer, compute_rerank_score, TRUST_SCORE_THRESHOLD
from litellm import completion
from functools import partial

class LanguageModel:
    def __init__(self,
        model_name: str,
        enable_confidence: bool = True,
        confidence_n_samples: int = 3,
    ) -> None:
        """
        LanguageModel class to interact with different language models.

        Arguments:
            model_name : str : The name of the language model to use.
            enable_confidence : bool : Whether to enable confidence scoring (default: True).
            confidence_n_samples : int : Number of ensemble samples for confidence (default: 3).

        Raises:
            ValueError : If the model name is not found.
        """

        self.model_name = model_name
        self.enable_confidence = enable_confidence
        self.confidence_n_samples = confidence_n_samples

        # Load the client for the model based on the model name
        if self.model_name in [
            "openai/gpt-4o-mini", "openai/gpt-4o-mini-2024-07-18",
            "openai/gpt-4o", "openai/gpt-4o-2024-08-06", "openai/gpt-4o-2024-11-20",
            "openai/gpt-3.5-turbo",
            "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "openai/o3-mini", "openai/o3-mini-2025-01-31",
            "openai/o1", "openai/o1-2024-12-17",
            "anthropic/claude-3-5-sonnet-latest", "anthropic/claude-3-5-sonnet-20241022",
            "anthropic/claude-3-5-haiku-latest", "anthropic/claude-3-5-haiku-20241022",
            "anthropic/claude-3-7-sonnet-latest", "anthropic/claude-3-7-sonnet-20250219",
            "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "together_ai/deepseek-ai/DeepSeek-R1",
            "together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            "together_ai/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "together_ai/Qwen/Qwen2.5-Coder-32B-Instruct",
            "together_ai/Qwen/QwQ-32B",
            "together_ai/Qwen/Qwen2-72B-Instruct",
            "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
            "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
            "gemini/gemini-2.0-flash",
            "ollama/llama3:70b",
        ]:
            self.client = partial(completion, model=self.model_name)
        else:
            raise ValueError(f"Model '{model_name}' not found.")

        self.gpt4Tokenizer = tiktoken.encoding_for_model('gpt-4o')

        # Initialize confidence scorer if enabled
        if self.enable_confidence:
            self.confidence_scorer = ConfidenceScorer(
                model_client=self.client,
                n_samples=self.confidence_n_samples
            )
            self.confidence_metadata = {}  # Cache for retrieval, indexed by example number
        

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.
        """
        tokens = self.gpt4Tokenizer.encode(text)
        return len(tokens)

    def _compute_and_store_confidence(self,
                                      history: List[dict],
                                      generator_answer: str,
                                      max_tokens: int = 2048) -> dict:
        """
        Compute confidence score for a generation and store in metadata cache.

        Arguments:
            history : List[dict] : The conversation history used for generation.
            generator_answer : str : The extracted answer from the generation.
            max_tokens : int : Maximum tokens for ensemble generation.

        Returns:
            dict : Confidence result containing trust_score, ensemble_answers, etc.
        """
        if not self.enable_confidence:
            return None

        try:
            confidence_result = self.confidence_scorer.compute_trust_score(
                prompt=history,
                original_answer=generator_answer,
                max_tokens=max_tokens
            )

            # Cache trust score for future retrieval (indexed by example number)
            example_idx = len(self.confidence_metadata)
            self.confidence_metadata[example_idx] = confidence_result

            return confidence_result
        except Exception as e:
            print(f"Warning: Failed to compute confidence score: {e}")
            return None

    def generate(self,
        history: List[str],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        current_depth: int = 1,
        max_depth_num_rounds: int = 3,
        allow_code_execution: bool = True,
        code_execution_flag: str = "EXECUTE CODE!",
        final_output: str = ""
    ) -> str:
        """
        Generate a response from the language model.

        Arguments:
            history : List[str] : The conversation history.
            temperature : float : The sampling temperature for the model.
            max_tokens : int : The maximum number of tokens to generate.
            current_depth : int : The current depth of the conversation.
            max_depth_num_rounds : int : The maximum number of rounds allowed.
            allow_code_execution : bool : Whether to allow code execution.
            code_execution_flag : str : The flag to trigger code execution.
            final_output : str : The final output to return.

        Returns:
            str : The final output of the conversation.

        Raises:
            ValueError : If the history is empty.
        """
        if len(history) == 0:
            raise ValueError("History must contain at least one message.")
        

        # Generate the response from the language model
        output = self.client(
            messages=history,
            model=self.model_name,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        ).choices[0].message["content"]

        # If Python code execution is allowed, execute the code
        pre_code_execution_flag = output.split(code_execution_flag)[0].strip()
        if allow_code_execution and code_execution_flag in output and '```' == pre_code_execution_flag[-3:]:
            if code_execution_flag in output:
                output_prefix = output.split(code_execution_flag)[0].strip()
            else:
                # TODO (msuzgun): This is a temporary solution. We may want to find a better way to handle this.
                output_prefix = output
            executed_code = extract_and_run_python_code(output_prefix)
            executed_code = executed_code.strip()
            current_output = f"{output_prefix}\n{code_execution_flag}\n\n{executed_code}"
            final_output = f"{final_output}\n\n{current_output}".strip()
            # import pdb; pdb.set_trace()
            # print(f"*** This code has been executed:\n{executed_code}\n\n")
            # print(f"***And the output is:\n{current_output}")
            # If the current depth is less than or equal to the maximum depth, add a new message to the history
            if current_depth <= max_depth_num_rounds:
                warning_txt = ""
                if current_depth == max_depth_num_rounds:
                    warning_txt = f" (This is the last round. No more code execution will be allowed. Please present your final solution now.)"
                new_messages = [
                    {"role": "assistant", "content": current_output},
                    {"role": "user", "content": f"Proceed with any additional steps required and provide the completed solution. If everything is already complete, type FINAL ANSWER and submit it in the expected format. If you are stucked, please try alternative methods to solve the problem and provide the final solution.{warning_txt}"}
                ]
                history += new_messages
                return self.generate(
                    history=history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    current_depth=current_depth+1,
                    max_depth_num_rounds=max_depth_num_rounds,
                    allow_code_execution=allow_code_execution,
                    code_execution_flag=code_execution_flag,
                    final_output=final_output,
                )
            else:
                final_output = f"{final_output}\n\n{current_output}".strip()
                return final_output
        else:
            # If code execution is not allowed or no code block is found, return the final output
            # Add the output to the final output
            final_output = f"{final_output}\n\n{output}".strip()
            return final_output

    def advanced_generate(self,
        approach_name: str,
        input_txt: str,
        cheatsheet: str = None,
        generator_template: str = None,
        cheatsheet_template: str = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_num_rounds: int = 1,
        allow_code_execution: bool = True,
        code_execution_flag: str = "EXECUTE CODE!",
        add_previous_answers_to_cheatsheet: bool = True,
        original_input_corpus: List[str] = None,
        original_input_embeddings: np.ndarray = None,
        generator_outputs_so_far: List[str] = None,
        retrieve_top_k: int = 3,
    ) -> Tuple[str, str, str, str]:
        """
        Generate a response from the language model.

        Arguments:
            approach_name : str : The name of the approach to use.
            input_txt : str : The input text for the model.
            cheatsheet : str : The cheatsheet for the model.
            generator_template : str : The template for the generator model.
            cheatsheet_template : str : The template for the cheatsheet extraction model.
            temperature : float : The sampling temperature for the model.
            max_tokens : int : The maximum number of tokens to generate.
            max_num_rounds : int : The maximum number of rounds allowed.
            allow_code_execution : bool : Whether to allow code execution.
            code_execution_flag : str : The flag to trigger code execution.
            add_previous_answers_to_cheatsheet : bool : Whether to add the previous answers to the cheatsheet.
            original_input_corpus : List[str] : The original input corpus.
            original_input_embeddings : np.ndarray : The original input embeddings.
            generator_outputs_so_far : List[str] : The generator outputs so far.
            retrieve_top_k : int : The number of top k inputs to retrieve.

        Returns:
            Tuple[str, str, str, str] : The generator answer, evaluator solution, answer check, and new cheatsheet.

        Raises:
            ValueError : If the proper templates are not provided.
        """

        # If the approach name is "default", run the generator model with the input text and the current cheatsheet
        if approach_name == "default":
            generator_prompt = generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", "(empty)")
            generator_history = [
                {"role": "user", "content": generator_prompt},
            ]
            generator_output = self.generate(
                history=generator_history,
                temperature=temperature,
                max_tokens=max_tokens,
                allow_code_execution=allow_code_execution,
                code_execution_flag=code_execution_flag,
            )

            generator_answer = extract_answer(
                generator_output,
            )

            

            output_dict = {
                "input_txt": input_txt,
                "steps": [
                    {
                        "round": 0,
                        "generator_prompt": generator_prompt,
                        "generator_output": generator_output,
                        "generator_answer": generator_answer,
                        "current_cheatsheet": None,
                        "new_cheatsheet": None,
                    }
                ],
                "previous_answers": None,
                "final_answer": generator_answer,
                "final_output": generator_output,
                "final_cheatsheet": None,
                "generator_output": generator_output,
            }

            if self.enable_confidence:
                # Compute confidence for this generation
                confidence_result = self._compute_and_store_confidence(
                    history=generator_history,
                    generator_answer=generator_answer,
                    max_tokens=max_tokens
                )
                if confidence_result is not None:
                    output_dict["confidence"] = confidence_result

            return output_dict
        
        elif approach_name == "DynamicCheatsheet_Cumulative":
            if cheatsheet is None:
                raise ValueError("Cheatsheet must be provided for dynamic_cheatsheet approach.")
            if cheatsheet_template is None:
                raise ValueError("Cheatsheet template must be provided for dynamic_cheatsheet approach.")
            
            steps = []
            previous_answers = []

            generator_output = ''

            for round in range(max(1, max_num_rounds)):
                ## STEP 1: Run the generator model with the input text and the cheatsheet
                generator_cheatsheet_content = cheatsheet

                # If there are previous answers, add them to the cheatsheet content for the generator
                if round > 0 and add_previous_answers_to_cheatsheet:
                    previous_answers_txt = f"PREVIOUS ANSWERS:\n{'; '.join(previous_answers)}"
                    generator_cheatsheet_content = f"{generator_cheatsheet_content}\n\n{previous_answers_txt}"

                generator_prompt = generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", generator_cheatsheet_content)
                current_cheatsheet = cheatsheet

                # Prepare the message history for the generator model
                generator_history = [{"role": "user", "content": generator_prompt}]
                # Run the generator model
                generator_output = self.generate(
                    history=generator_history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    allow_code_execution=allow_code_execution,
                    code_execution_flag=code_execution_flag,
                )
                # Extract the output from the generator model
                generator_answer = extract_answer(generator_output)

                ## STEP 2: Run the cheatsheet extraction model with the generator output and the current cheatsheet
                cheatsheet_prompt = cheatsheet_template.replace("[[QUESTION]]", input_txt).replace("[[MODEL_ANSWER]]", generator_output).replace("[[PREVIOUS_CHEATSHEET]]", current_cheatsheet)

                cheatsheet_history = [{"role": "user", "content": cheatsheet_prompt}]
                cheatsheet_output = self.generate(
                    history=cheatsheet_history,
                    temperature=temperature,
                    max_tokens=2*max_tokens,
                    allow_code_execution=False,
                )

                # Extract the new cheatsheet from the output (if present); otherwise, return the old cheatsheet
                new_cheatsheet = extract_cheatsheet(response=cheatsheet_output, old_cheatsheet=current_cheatsheet)
                cheatsheet = new_cheatsheet

                previous_answers.append(f"Round {round+1}: {generator_answer}")
            
                steps.append({
                    "round": round,
                    "generator_prompt": generator_prompt,
                    "generator_output": generator_output,
                    "generator_answer": generator_answer,
                    "current_cheatsheet": current_cheatsheet,
                    "new_cheatsheet": new_cheatsheet,
                })

            output_dict = {
                "input_txt": input_txt,
                "steps": steps,
                "previous_answers": previous_answers,
                "final_answer": generator_answer,
                "final_cheatsheet": new_cheatsheet,
                "final_output": generator_output,
            }

            if self.enable_confidence:
                # Compute confidence for this generation
                confidence_result = self._compute_and_store_confidence(
                    history=generator_history,
                    generator_answer=generator_answer,
                    max_tokens=max_tokens
                )
                if confidence_result is not None:
                    output_dict["confidence"] = confidence_result

            return output_dict

        elif approach_name == "FullHistoryAppending":
            length_of_history = len(generator_outputs_so_far)
            if length_of_history > 0:
                top_k_original_inputs = original_input_corpus[:length_of_history]
                top_k_original_outputs = generator_outputs_so_far

                curated_cheatsheet = "### PREVIOUS SOLUTIONS (START)\n\n"
                for i, (previous_input_txt, previous_output_txt) in enumerate(zip(original_input_corpus, generator_outputs_so_far)):
                    curated_cheatsheet += f"#### Previous Input #{i+1}:\n\n{previous_input_txt}\n\n#### Model Solution to Previous Input #{i+1}:\n\n{previous_output_txt}\n---\n---\n\n"
                curated_cheatsheet += "#### PREVIOUS SOLUTIONS (END)"
            else:
                top_k_original_inputs = []
                top_k_original_outputs = []
                curated_cheatsheet = "(empty)"
            
            # Replace the relevant placeholders in the generator template with the input text and the curated cheatsheet and then run the generator model
            generator_prompt = generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", curated_cheatsheet)
            generator_history = [{"role": "user", "content": generator_prompt}]
            generator_output = self.generate(
                    history=generator_history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    allow_code_execution=allow_code_execution,
                    code_execution_flag=code_execution_flag,
                )
            # Extract the answer from the generator model
            generator_answer = extract_answer(generator_output)

            return {
                "input_txt": input_txt,
                "steps": [
                    {
                        "round": 0,
                        "generator_prompt": generator_prompt,
                        "generator_output": generator_output,
                        "generator_answer": generator_answer,
                        "current_cheatsheet": curated_cheatsheet,
                        "new_cheatsheet": None,
                    }
                ],
                "top_k_original_inputs": top_k_original_inputs,
                "top_k_original_outputs": top_k_original_outputs,
                "final_answer": generator_answer,
                "final_output": generator_output,
                "final_cheatsheet": curated_cheatsheet,
            }
        elif approach_name in ["Dynamic_Retrieval", "DynamicCheatsheet_RetrievalSynthesis"]:
            # Get the current original input embedding
            current_original_input_embedding = original_input_embeddings[-1] # Current original input embedding
            prev_original_input_embeddings = original_input_embeddings[:-1] # Note that this can be empty
            
            # Retrieve the most similar k input-output pairs from the previous inputs and outputs
            if len(prev_original_input_embeddings) > 0:
                # Compute cosine similarities
                similarities = cosine_similarity([current_original_input_embedding], prev_original_input_embeddings)
                top_k_indices = np.argsort(similarities[0])[::-1][:retrieve_top_k]

                if self.enable_confidence:
                    # Get trust scores from confidence metadata cache
                    trust_scores = []
                    for i in range(len(prev_original_input_embeddings)):
                        confidence_data = self.confidence_metadata.get(i, {})
                        trust_scores.append(confidence_data.get('trust_score', 0.5))  # Default 0.5 if not available

                    # Compute recency scores: exp(-0.1 * (N - i))
                    N = len(prev_original_input_embeddings)
                    recency_scores = [np.exp(-0.1 * (N - i)) for i in range(N)]

                    # Compute diversity scores (simple uniform version for now)
                    diversity_scores = [1.0] * N

                    # Compute rerank scores combining all factors
                    rerank_scores = []
                    for i in range(N):
                        rerank_score = compute_rerank_score(
                            similarity=similarities[0][i],
                            trust_score=trust_scores[i],
                            recency=recency_scores[i],
                            diversity=diversity_scores[i]
                        )
                        rerank_scores.append(rerank_score)

                    # Use rerank_scores instead of similarities for top-k selection
                    top_k_indices = np.argsort(rerank_scores)[::-1][:retrieve_top_k]

                    # Filter out retrieved items with trust_score below threshold
                    top_k_indices = [i for i in top_k_indices if trust_scores[i] >= TRUST_SCORE_THRESHOLD]

                top_k_original_inputs = [original_input_corpus[i] for i in top_k_indices]
                top_k_original_outputs = [generator_outputs_so_far[i] for i in top_k_indices]
                top_k_similar_values = similarities[0][top_k_indices]
                # Use the retrieved pairs to curate the cheatsheet for the generator model
                curated_cheatsheet = "### PREVIOUS SOLUTIONS (START)\n\nNote: The input-output pairs listed below are taken from previous test cases and are meant to assist you in understanding potential solution strategies or tool usages. While they can offer insight and inspiration, they should not be blindly copied, as they may contain errors or may not fit your specific use case. Approach them with a critical mindset—analyze their logic, verify their correctness, and adapt them as needed. Your goal should be to develop a well-reasoned solution that best addresses the problem at hand.\n\n"
            else:
                top_k_original_inputs = []
                top_k_original_outputs = []
                top_k_similar_values = []
                curated_cheatsheet = '(empty)'
            
            # The following only adds the previous input-output pairs to the cheatsheet
            for i, (previous_input_txt, previous_output_txt, similarity) in enumerate(zip(top_k_original_inputs[::-1], top_k_original_outputs[::-1], top_k_similar_values[::-1])):
                curated_cheatsheet += f"#### Previous Input #{i+1} (Similarity: {similarity:.2f}):\n\n{previous_input_txt}\n\n#### Model Solution to Previous Input  #{i+1}:\n\n{previous_output_txt}\n---\n---\n\n"
            curated_cheatsheet = curated_cheatsheet.strip()
            
            # If it is empty, we should not add the "PREVIOUS SOLUTIONS (END)" to the cheatsheet
            if curated_cheatsheet != '(empty)':
                curated_cheatsheet += "\n\n#### PREVIOUS SOLUTIONS (END)"

            # Run the Generator model with the input text and the curated cheatsheet (input-output pairs) to generate a better (more tailored) cheatsheet   
            previous_cheatsheet = cheatsheet
            if approach_name == "DynamicCheatsheet_RetrievalSynthesis":
                # First, we need to make the necessary replacements in the cheatsheet template
                cheatsheet_prompt = cheatsheet_template.replace("[[PREVIOUS_INPUT_OUTPUT_PAIRS]]", curated_cheatsheet)
                cheatsheet_prompt = cheatsheet_prompt.replace("[[NEXT_INPUT]]", input_txt)
                cheatsheet_prompt = cheatsheet_prompt.replace("[[PREVIOUS_CHEATSHEET]]", previous_cheatsheet)
                # Now, we are ready to run the cheatsheet curator model
                cheatsheet_history = [{"role": "user", "content": cheatsheet_prompt}]
                cheatsheet_output = self.generate(
                    history=cheatsheet_history,
                    temperature=temperature,
                    max_tokens=2*max_tokens,
                    allow_code_execution=False,
                )
                # Finally, extract the new cheatsheet from the output (if present); otherwise, return the old cheatsheet
                new_cheatsheet = extract_cheatsheet(response=cheatsheet_output, old_cheatsheet=curated_cheatsheet)
                curated_cheatsheet = new_cheatsheet

            # Replace the relevant placeholders in the generator template with the input text and the curated cheatsheet and then run the generator model
            generator_prompt = generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", curated_cheatsheet)
            generator_history = [{"role": "user", "content": generator_prompt}]
            generator_output = self.generate(
                    history=generator_history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    allow_code_execution=allow_code_execution,
                    code_execution_flag=code_execution_flag,
                )
            # Extract the answer from the generator model
            generator_answer = extract_answer(generator_output)

            output_dict = {
                "input_txt": input_txt,
                "steps": [
                    {
                        "round": 0,
                        "generator_prompt": generator_prompt,
                        "generator_output": generator_output,
                        "generator_answer": generator_answer,
                        "current_cheatsheet": curated_cheatsheet,
                        "new_cheatsheet": None,
                    }
                ],
                "top_k_original_inputs": top_k_original_inputs,
                "top_k_original_outputs": top_k_original_outputs,
                "final_answer": generator_answer,
                "final_output": generator_output,
                "final_cheatsheet": curated_cheatsheet,
            }

            if self.enable_confidence:
                # Compute confidence for this generation
                confidence_result = self._compute_and_store_confidence(
                    history=generator_history,
                    generator_answer=generator_answer,
                    max_tokens=max_tokens
                )
                if confidence_result is not None:
                    output_dict["confidence"] = confidence_result

            return output_dict
        else:
            raise ValueError(f"Approach '{approach_name}' not found.")