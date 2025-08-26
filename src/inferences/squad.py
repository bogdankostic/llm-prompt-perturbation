import argparse
import json
from collections import defaultdict
import statistics

from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.evaluators import SASEvaluator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from datasets import load_dataset
import evaluate

from src.utils.experiment import Experiment
from src.custom_components.generator.cached_chat_generator import CachedOpenAIChatGenerator


system_prompt = """You are a helpful assistant for extractive question answering. Read the provided context and answer the question with the shortest accurate span from the context.
Reply with the answer only, do not include any other text.
"""
prompt_template = """Context: {{context}}
Question: {{question}}
Answer:
"""

chat_messages = [
    ChatMessage.from_system(system_prompt),
    ChatMessage.from_user(prompt_template)
]


def main():
    parser = argparse.ArgumentParser(description="Run SQuAD experiments with optionally perturbed data")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the (perturbed) dataset JSON")
    parser.add_argument("--perturbation-type", type=str, default="none", help="Type of perturbation")
    parser.add_argument("--model", type=str, required=True, help="Model to use for generation")
    parser.add_argument("--api-base-url", type=str, default=None, help="API base URL for the model")
    parser.add_argument("--output", type=str, required=True, help="Output path for the experiment results")
    args = parser.parse_args()

    # Load dataset
    squad_dataset = load_dataset("rajpurkar/squad", split="validation").shuffle(seed=77).select(range(1000))
    if args.perturbation_type != "none":
        squad_dataset = squad_dataset.remove_columns(["context", "question"])
        with open(args.dataset, "r") as f:
            data = json.load(f)["data"]
        squad_dataset = squad_dataset.add_column("context", [item["context"] for item in data])
        squad_dataset = squad_dataset.add_column("question", [item["question"] for item in data])

    # Build pipeline
    prompt_builder = ChatPromptBuilder(template=chat_messages)
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

    pipeline = Pipeline()
    pipeline.add_component(name="prompt_builder", instance=prompt_builder)
    pipeline.add_component(name="generator", instance=generator)
    pipeline.connect("prompt_builder", "generator")
    squad_evaluator = evaluate.load("squad")
    sas_evaluator = SASEvaluator()

    # Create experiment record
    experiment = Experiment(
        name=f"squad_{args.model}_{args.perturbation_type}_perturbation",
        configs={"pipeline": pipeline.to_dict()},
        dataset="squad",
        model=args.model,
        perturbation_type=args.perturbation_type,
    )

    # Inference loop
    predictions = defaultdict(list)
    total_samples = len(squad_dataset)
    em_scores = []
    f1_scores = []
    sas_scores = []
    for idx, item in enumerate(squad_dataset):
        print(f"Processing sample {idx + 1}/{total_samples}")
        response = pipeline.run({
            "question": item.get("question", ""),
            "context": item.get("context", ""),
        })
        model_answer = response["generator"]["replies"][0].text
        ground_truths = list(set(item.get("answers", {}).get("text", [])))

        predictions["id"].append(item.get("id"))
        predictions["question"].append(item.get("question", ""))
        predictions["context"].append(item.get("context", ""))
        predictions["answers"].append(ground_truths)
        predictions["prediction"].append(model_answer)
        predictions["output"].append(response["generator"]["replies"][0].text)

        squad_eval = squad_evaluator.compute(
            predictions=[{"id": item.get("id"), "prediction_text": model_answer}],
            references=[{"id": item.get("id"), "answers": item.get("answers")}])
        sas_eval = sas_evaluator.run(ground_truths, [model_answer] * len(ground_truths))
        em_scores.append(squad_eval["exact_match"])
        f1_scores.append(squad_eval["f1"])
        sas_scores.append(max(sas_eval["individual_scores"]))

    experiment.add_predictions(predictions)

    # Metrics
    experiment.add_metrics({
        "exact_match": statistics.mean(em_scores),
        "f1": statistics.mean(f1_scores),
        "sas": statistics.mean(sas_scores),
    })

    experiment.save(args.output)


if __name__ == "__main__":
    main()
