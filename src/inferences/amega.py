import argparse
import json
from collections import defaultdict
import subprocess
import statistics

from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
import pandas as pd

from src.utils.experiment import Experiment
from src.custom_components.generator.cached_chat_generator import CachedOpenAIChatGenerator
from src.custom_components.evaluator.amega import AMEGAEvaluator


model_prompt_template = """Initial Case: {{case_text}}
Question: {{question}}
"""
evaluator_prompt_template = """Evaluate the text below given the criteria list.
Therefore, return a list of True or False for each criterion, depending on whether the text below meets this criterion or not.
Do not evalaute each bullet point of the text separately. Do not justify your decision.
Text: {{candidate_response}}
Criteria: {{criteria_list}}
"""

model_chat_messages = [ChatMessage.from_user(model_prompt_template)]
evaluator_chat_messages = [ChatMessage.from_user(evaluator_prompt_template)]


def main():
    parser = argparse.ArgumentParser(description="Run SQuAD experiments with optionally perturbed data")
    parser.add_argument("--cases", type=str, required=True, help="Path to the (perturbed) cases data")
    parser.add_argument("--questions", type=str, required=True, help="Path to the (perturbed) question data")
    parser.add_argument("--criteria", type=str, required=True, help="Path to the criteria data")
    parser.add_argument("--perturbation-type", type=str, default="none", help="Type of perturbation")
    parser.add_argument("--model", type=str, required=True, help="Model to use for generation")
    parser.add_argument("--api-base-url", type=str, default=None, help="API base URL for the model")
    parser.add_argument("--output", type=str, required=True, help="Output path for the experiment results")
    args = parser.parse_args()

    # Clone the AMEGA dataset
    subprocess.run(["git", "clone", "https://github.com/DATEXIS/AMEGA-benchmark.git"])

    # Load dataset
    criteria_df = pd.read_csv("AMEGA-benchmark/data/criteria.csv", sep=";")
    if args.perturbation_type == "none":
        cases_df = pd.read_csv("AMEGA-benchmark/data/cases.csv", sep=";")
        questions_df = pd.read_csv("AMEGA-benchmark/data/questions.csv", sep=";")
    else:
        with open(args.cases, "r") as f:
            cases_data = json.load(f)["data"]
            cases_df = pd.DataFrame(cases_data)
        with open(args.questions, "r") as f:
            questions_data = json.load(f)["data"]
            questions_df = pd.DataFrame(questions_data)

    # Build model pipeline
    model_prompt_builder = ChatPromptBuilder(template=model_chat_messages)
    generator = CachedOpenAIChatGenerator(
        model=args.model,
        api_key=Secret.from_env_var("OPENAI_API_KEY") if args.api_base_url is None else Secret.from_env_var("PLACEHOLDER"),
        cache_dir="/experiments/llm-cache",
        api_base_url=args.api_base_url,
        generation_kwargs={
            "temperature": 0,
            "seed": 77,
        },
    )

    model_pipeline = Pipeline()
    model_pipeline.add_component(name="prompt_builder", instance=model_prompt_builder)
    model_pipeline.add_component(name="generator", instance=generator)
    model_pipeline.connect("prompt_builder", "generator")

    # Initialize evaluator
    evaluator = AMEGAEvaluator()

    # Create experiment record
    experiment = Experiment(
        name=f"amega_{args.model}_{args.perturbation_type}_perturbation",
        configs={"pipeline": model_pipeline.to_dict()},
        dataset="amega",
        model=args.model,
        perturbation_type=args.perturbation_type,
    )

    # Inference loop
    predictions = defaultdict(list)
    total_samples = len(questions_df)
    for idx, row in questions_df.iterrows():
        print(f"Processing sample {idx + 1}/{total_samples}")
        case_id = row.get("case_id")
        question_id = row.get("question_id")
        case_text = cases_df.loc[cases_df["case_id"] == case_id, "case_str"].values[0]
        question_text = row.get("question_str")

        model_response = model_pipeline.run({
            "case_text": case_text,
            "question": question_text,
        })
        model_answer = model_response["generator"]["replies"][0].text

        # Evaluate the model answer
        criteria_list = criteria_df.loc[(criteria_df["case_id"] == case_id) & (criteria_df["question_id"] == question_id), "criteria_str"].values.tolist()
        evaluator_response = evaluator.run(model_answer, criteria_list)
        majority_vote = evaluator_response["majority_vote"]
        true_rates = evaluator_response["true_rates"]
        confidence_rate = evaluator_response["confidence_rate"]
        fail_rate = evaluator_response["fail_rate"]

        predictions["case_id"].append(case_id)
        predictions["question_id"].append(question_id)
        predictions["model_answer"].append(model_answer)
        predictions["majority_vote"].append(majority_vote)
        predictions["true_rates"].append(true_rates)
        predictions["mean_true_rate"].append(statistics.mean(true_rates))
        predictions["confidence_rate"].append(confidence_rate)
        predictions["fail_rate"].append(fail_rate)

    experiment.add_predictions(predictions)

    experiment.add_metrics({
        "mean_true_rate": statistics.mean(predictions["mean_true_rate"]),
        "confidence_rate": statistics.mean(predictions["confidence_rate"]),
        "fail_rate": statistics.mean(predictions["fail_rate"]),
    })

    experiment.save(args.output)


if __name__ == "__main__":
    main()
