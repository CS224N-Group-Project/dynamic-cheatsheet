"""
This module provides confidence scoring functionality using ensemble agreement.

The module implements a multi-factor confidence scoring system that:
1. Uses ensemble agreement as the primary trust signal
2. Optionally includes answer parseability and code executability checks
3. Provides reranking functions for retrieval based on similarity, trust, recency, and diversity

Classes:
* ConfidenceScorer: Computes trust scores using ensemble agreement without scalar self-assessment.

Functions:
* compute_rerank_score: Combines multiple factors for retrieval reranking.
"""

from typing import List, Dict
from collections import Counter
from .extractor import extract_answer
from .execute_code import extract_and_run_python_code


class ConfidenceScorer:
    """
    Computes trust scores using ensemble agreement.

    This scorer generates multiple responses at different temperatures and measures
    agreement as a proxy for confidence. Higher agreement indicates higher confidence
    in the answer.
    """

    def __init__(self,
                 model_client,
                 n_samples: int = 3,
                 temperatures: List[float] = None):
        """
        Args:
            model_client: The language model client for generating responses 
            n_samples: Number of ensemble samples to generate (default: 3)
            temperatures: Temperature values for sampling diversity (default: [0.7, 0.8, 0.9])
        """
        self.model_client = model_client
        self.n_samples = n_samples
        self.temperatures = temperatures if temperatures is not None else [0.7, 0.8, 0.9]

        if len(self.temperatures) < n_samples:
            self.temperatures = self.temperatures + [self.temperatures[-1]] * (n_samples - len(self.temperatures))

    def compute_trust_score(self,
                           prompt: List[dict],
                           original_answer: str,
                           max_tokens: int = 2048) -> Dict:
        """
        Generates multiple responses and computes agreement-based trust score (can use multiple signals).

        Args:
            prompt: The conversation history/prompt (list of message dicts)
            original_answer: The answer from the original generation
            max_tokens: Maximum tokens for generation (default: 2048)

        Returns:
            Dict containing:
                - trust_score: float (0-1), the agreement score
                - ensemble_answers: List[str], all extracted answers
                - agreement: float (0-1), same as trust_score
                - most_common_answer: str, the most frequent answer
                - signals: Dict with breakdown of individual signals
        """
        ensemble_result = self.compute_ensemble_agreement(
            prompt=prompt,
            max_tokens=max_tokens
        )

        signals = {
            'ensemble_agreement': ensemble_result['agreement']
        }

        # Optional answer parseability check (currently not used)
        answer_parseable = 1.0 if original_answer != "No final answer found" else 0.0
        signals['answer_parseable'] = answer_parseable

        trust_score = ensemble_result['agreement']

        return {
            'trust_score': trust_score,
            'ensemble_answers': ensemble_result['ensemble_answers'],
            'agreement': ensemble_result['agreement'],
            'most_common_answer': ensemble_result['most_common_answer'],
            'signals': signals
        }

    def compute_ensemble_agreement(self,
                                  prompt: List[dict],
                                  max_tokens: int = 2048) -> Dict:
        """
        Generate N responses with different temperatures and compute agreement.

        Algorithm:
        1. Generate N responses with different temperatures
        2. Extract answer from each response
        3. Find most common answer and compute agreement ratio

        Args:
            prompt: The conversation history/prompt (list of message dicts)
            max_tokens: Maximum tokens for generation

        Returns:
            Dict containing:
                - agreement: float (0-1), ratio of most common answer
                - ensemble_answers: List[str], all extracted answers
                - most_common_answer: str, the most frequent answer
                - ensemble_responses: List[str], all raw responses
        """
        ensemble_responses = []

        # Generate N responses with different temperatures
        for i in range(self.n_samples):
            temp = self.temperatures[i]
            try:
                response = self.model_client(
                    messages=prompt,
                    temperature=temp,
                    max_completion_tokens=max_tokens,
                    # Note: We don't execute code for ensemble samples (too expensive)
                )
                response_text = response.choices[0].message["content"]
                ensemble_responses.append(response_text)
            except Exception as e:
                print(f"Warning: Failed to generate ensemble sample {i+1} with temp={temp}: {e}")
                ensemble_responses.append("No final answer found")

        ensemble_answers = [extract_answer(resp) for resp in ensemble_responses]
        answer_counts = Counter(ensemble_answers)

        if len(answer_counts) > 0:
            most_common_answer, max_count = answer_counts.most_common(1)[0]
        else:
            most_common_answer = "No final answer found"
            max_count = 0

        agreement = max_count / len(ensemble_answers) if len(ensemble_answers) > 0 else 0.0

        return {
            'agreement': agreement,
            'ensemble_answers': ensemble_answers,
            'most_common_answer': most_common_answer,
            'ensemble_responses': ensemble_responses
        }


def compute_rerank_score(similarity: float,
                        trust_score: float,
                        recency: float,
                        diversity: float,
                        weights: Dict = None) -> float:
    """
    Compute a combined reranking score from multiple factors.

    This function combines similarity, trust, recency, and diversity scores
    using a weighted sum. All input scores should be normalized to [0, 1].

    Args:
        similarity: Cosine similarity score (0-1)
        trust_score: Ensemble agreement trust score (0-1)
        recency: Recency score, typically exp(-0.1 * (N - i)) (0-1)
        diversity: Diversity score (0-1)
        weights: Optional dict of weights for each factor
                 Default: {'similarity': 0.4, 'trust': 0.3, 'recency': 0.15, 'diversity': 0.15}

    Returns:
        float: Combined reranking score (0-1)
    """
    if weights is None:
        weights = {
            'similarity': 0.4,
            'trust': 0.3,
            'recency': 0.15,
            'diversity': 0.15
        }

    # Compute weighted sum
    score = (
        weights['similarity'] * similarity +
        weights['trust'] * trust_score +
        weights['recency'] * recency +
        weights['diversity'] * diversity
    )

    return score
