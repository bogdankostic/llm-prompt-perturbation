import argparse
import json
from collections import defaultdict

from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage

from src.utils.experiment import Experiment
from src.utils.metrics import exact_match
from src.custom_components.generator.cached_chat_generator import CachedOpenAIChatGenerator


answer_mapping = ["A", "B", "C", "D"]

system_prompt = """You are a helpful assistant that can answer multiple choice questions about the world. Answer the question based on the choices provided. Only output the letter of the choice that you think is the correct answer."""
prompt_template = """Question: {{question}}
{% for choice in choices %}{{ ['A', 'B', 'C', 'D'][loop.index0] }}) {{choice}}
{% endfor %}
Answer:
"""

chat_messages = [
    ChatMessage.from_system(system_prompt),
    ChatMessage.from_user(prompt_template)
]

def main():
    parser = argparse.ArgumentParser(description="Run MMLU experiments with pre-generated lexical perturbations")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the (perturbed) dataset")
    parser.add_argument("--perturbation-type", type=str, default="none", help="Type of perturbation")
    parser.add_argument("--model", type=str, required=True, help="Model to use for generation")
    parser.add_argument("--api-base-url", type=str, default=None, help="API base URL for the model")
    parser.add_argument("--output", type=str, required=True, help="Output path for the experiment results")
    args = parser.parse_args()

    # Load data
    data_path = args.dataset
    with open(data_path, "r") as f:
        data = json.load(f)

    # Create pipeline components
    prompt_builder = ChatPromptBuilder(template=chat_messages)
    generation_kwargs = {"seed": 77}
    # Set reasoning effort to low for GPT-OSS models
    if args.model.startswith("openai/gpt-oss"):
        generation_kwargs["reasoning_effort"] = "low"
    # Set verbosity to low and reasoning effort to minimal to keep costs low
    if args.model.startswith("gpt-5"):
        generation_kwargs["verbosity"] = "low"
        generation_kwargs["reasoning_effort"] = "minimal"
    # New GPT-5 models do not support setting custom temperature value, so it's only set for other models
    else:
        generation_kwargs["temperature"] = 0
    generator = CachedOpenAIChatGenerator(
        model=args.model,
        api_key=Secret.from_env_var("OPENAI_API_KEY") if args.api_base_url is None else Secret.from_env_var("PLACEHOLDER"),
        cache_dir="/experiments/llm-cache",
        api_base_url=args.api_base_url,
        generation_kwargs=generation_kwargs,
    )

    # Create pipeline
    pipeline = Pipeline()
    pipeline.add_component(name="prompt_builder", instance=prompt_builder)
    pipeline.add_component(name="generator", instance=generator)
    pipeline.connect("prompt_builder", "generator")

    # Create experiment
    experiment = Experiment(
        name=f"mmlu_{args.model}_{args.perturbation_type}_perturbation",
        configs={"pipeline": pipeline.to_dict()},
        dataset="mmlu",
        model=args.model,
        perturbation_type=args.perturbation_type
    )

    # Run inference on LLM
    predictions = defaultdict(list)
    total_samples = len(data["data"])
    for idx, item in enumerate(data["data"]):
        print(f"Processing sample {idx + 1}/{total_samples}")
        response = pipeline.run({
            "question": item["question"],
            "choices": item["choices"]
        })
        predictions["dataset"].append(item["dataset"])
        predictions["question"].append(item["question"])
        # Normalize ground-truth answer to letter (A-D)
        true_answer = item["answer"]
        if isinstance(true_answer, int):
            true_answer = answer_mapping[true_answer]
        predictions["answer"].append(true_answer)

        # Normalize model prediction to a single uppercase letter (A-D)
        model_text = response["generator"]["replies"][0].text
        model_letter = str(model_text).strip()[:1].upper() if model_text else ""
        predictions["prediction"].append(model_letter)
        predictions["output"].append(response["generator"]["replies"][0].text)

    experiment.add_predictions(predictions)

    # Calculate Exact Match
    exact_match_score = exact_match(predictions["prediction"], predictions["answer"])

    # Add metrics to experiment
    experiment.add_metrics({
        "exact_match": exact_match_score,
    })

    experiment.save(args.output)


if __name__ == "__main__":
    main()
