import json
import re

from haystack import component
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses.chat_message import ChatMessage
from vertexai.generative_models import GenerativeModel, GenerationConfig
import numpy as np

from src.utils.rate_limiter import RateLimiter

PROMPT_TEMPLATE = """Evaluate the text below given the criteria list.
Therefore, return a list of True or False for each criterion, depending on whether the text below meets this criterion or not.
Do not evalaute each bullet point of the text separately. Do not justify your decision.
Text: {{candidate_response}}
Criteria: {{criteria_list}}
"""


@component
class AMEGAEvaluator:
    """
    A component that evaluates an LLM's answer for an AMEGA question against a given set of criteria.
    Adapted from https://github.com/DATEXIS/AMEGA-benchmark/blob/5a2e4cefcda54f6dbbb72319328376cb5f1ab6fb/main.py
    """

    def __init__(
        self,
        model: str = "models/gemini-2.5-flash-lite",
        n_eval: int = 8,
        calls_per_second: float = 2.0,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ):
        """
        Initialize the AMEGAEvaluator component.

        :param model: The model to use for evaluation.
        :param n_eval: The number of evaluations to generate per question and criteria list combination. Set to 11 in the AMEGA repository, but Gemini API has a limit of 8.
        :param calls_per_second: Maximum number of API calls per second.
        :param max_retries: Maximum number of retries for API calls.
        :param base_delay: Base delay for exponential backoff in seconds.
        """
        self.model = model
        self.n_eval = n_eval
        self.rate_limiter = RateLimiter(
            calls_per_second=calls_per_second,
            max_retries=max_retries,
            base_delay=base_delay
        )
        self.prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(PROMPT_TEMPLATE)], required_variables=["candidate_response", "criteria_list"])
        self.evaluator = GenerativeModel(
            model_name=self.model,
            generation_config=GenerationConfig(
                seed=77,
                candidate_count=self.n_eval,
                response_mime_type="application/json",
                response_schema={
                    "type": "array",
                    "items": {
                        "type": "boolean",
                    }
                }
            )
        )

    def _generate_with_retry(self, prompt_text: str):
        """
        Generate content with retry logic for rate limiting.
        """
        return self.rate_limiter.retry_with_backoff(
            self.evaluator.generate_content, 
            contents=prompt_text
        )

    @component.output_types(majority_vote=list[bool], mean_rates=list[float], confidence_rate=float, fail_rate=float)
    def run(self, model_response: str, criteria_list: list[str]):
        """
        Evaluate the model response against the criteria list.
        :param model_response: The model response to evaluate.
        :param criteria_list: The criteria list to evaluate the model response against.

        :return: A dictionary with the following keys
            - majority_vote: per-criterion majority decision across successful evaluator candidates; True if at least half returned True for that criterion.
            - true_rates: per-criterion share of True among successful evaluator candidates in [0.0, 1.0]. This is the empirical true rate used to derive the majority vote and confidence.
            - confidence_rate: agreement consistency in [0.0, 1.0] computed as 1 - 2 * mean(|round(p) - p|) over per-criterion true rates p. Equals 1.0 for unanimous agreement (all True or all False per criterion), and approaches 0.0 for maximal disagreement (~50/50).
            - fail_rate: fraction of evaluator candidates that produced unusable outputs (e.g., wrong length/format) in [0.0, 1.0].
        """
        result_matrix = np.full((self.n_eval, len(criteria_list)), np.nan)
        fail_rate = 1.0

        # Repeat the evaluation process until the fail rate is below 0.5
        while fail_rate > 0.5:
            criteria_json = json.dumps(criteria_list)
            evaluator_prompt_text = self.prompt_builder.run(candidate_response=model_response, criteria_list=criteria_json)["prompt"][0].text
            
            # Use retry logic for API call
            evaluator_model_response = self._generate_with_retry(evaluator_prompt_text)
            
            candidate_boolean_lists = [self._extract_booleans(candidate.content.parts[0].text) for candidate in evaluator_model_response.candidates if candidate.finish_reason == 1]
            
            if not candidate_boolean_lists:
                print("No valid evaluations! Repeat...")
                continue

            filtered_boolean_lists = [boolean_list for boolean_list in candidate_boolean_lists if len(boolean_list) == len(criteria_list)]
            result_matrix = np.array(filtered_boolean_lists)
            num_valid_evaluations = result_matrix.shape[0]
            fail_rate = self._calculate_fail_rate(num_valid_evaluations)

            # If the fail rate is higher than 0.5, we split the criteria list in half and evaluate each part separately
            if fail_rate > 0.5 and len(criteria_list) > 1:
                midpoint_index = len(criteria_list) // 2
                result_first = self.run(model_response, criteria_list[:midpoint_index])
                result_second = self.run(model_response, criteria_list[midpoint_index:])

                # Combine both halves
                combined_majority_vote = np.concatenate([result_first["majority_vote"], result_second["majority_vote"]])
                combined_true_rates = np.concatenate([result_first["true_rates"], result_second["true_rates"]])
                combined_confidence_rate = (result_first["confidence_rate"] + result_second["confidence_rate"]) / 2
                fail_rate = (result_first["fail_rate"] + result_second["fail_rate"]) / 2

                return {
                    "majority_vote": combined_majority_vote.tolist(),
                    "true_rates": combined_true_rates.tolist(),
                    "confidence_rate": combined_confidence_rate,
                    "fail_rate": np.round(fail_rate, 2),
                }
            
            if fail_rate <= 0.5:
                majority_vote, true_rates, confidence_rate = self._calculate_majority_vote(result_matrix)
                return {
                    "majority_vote": majority_vote.tolist(),
                    "true_rates": true_rates.tolist(),
                    "confidence_rate": confidence_rate,
                    "fail_rate": np.round(fail_rate, 2),
                }

        # Unexpected failure
        print("Unexpected failure!")
        return {
            "majority_vote": [np.nan] * len(criteria_list),
            "true_rates": [np.nan] * len(criteria_list),
            "confidence_rate": np.nan,
            "fail_rate": np.nan,
        }       

    @staticmethod
    def _extract_booleans(evaluator_response: str):
        """
        Extract boolean values from evaluator response string.
        """
        return [bool(re.match("true", boolean_occurrence)) for boolean_occurrence in re.findall(r"true|false", evaluator_response.lower())]
    
    def _calculate_fail_rate(self, num_valid_evaluations: int):
        """
        Calculate the fail rate based on the number of valid evaluations.
        """
        return 1 - (num_valid_evaluations / self.n_eval)
    
    def _calculate_majority_vote(self, result_matrix: np.ndarray):
        """
        Calculate the majority vote for each criterion.
        """
        criterion_true_rate = np.nansum(result_matrix, axis=0) / result_matrix.shape[0]
        majority_vote = criterion_true_rate >= 0.5
        confidence_rate = self._calculate_confidence_rate(criterion_true_rate)
        return majority_vote, criterion_true_rate, np.round(confidence_rate, 4)

    def _calculate_confidence_rate(self, criterion_true_rate: np.ndarray):
        """
        Calculate confidence rate as in the AMEGA reference: 1 - 2 * mean distance from nearest integer.
        """
        return 1 - 2 * np.sum(np.abs(np.round(criterion_true_rate) - criterion_true_rate)) / len(criterion_true_rate)
