from collections import defaultdict

from datasets import load_dataset, get_dataset_config_names
from haystack import Pipeline
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.utils import Secret
from tqdm import tqdm

from src.utils.experiment import Experiment
from src.utils.metrics import exact_match


answer_mapping = ["A", "B", "C", "D"]


prompt_template = """Question: {{question}}
{% for choice in choices %}{{ ['A', 'B', 'C', 'D'][loop.index0] }}) {{choice}}
{% endfor %}
Answer:
"""
system_prompt = """You are a helpful assistant that can answer multiple choice questions about the world. Answer the question based on the choices provided. Only output the letter of the choice that you think is the correct answer."""
prompt_builder = PromptBuilder(
    template=prompt_template
)
generator = OpenAIGenerator(
    model="gpt-4.1-nano-2025-04-14",
    system_prompt=system_prompt,
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    generation_kwargs={
        "temperature": 0,
        "seed": 77,
    }
)

pipeline = Pipeline()
pipeline.add_component(name="prompt_builder", instance=prompt_builder)
pipeline.add_component(name="generator", instance=generator)
pipeline.connect("prompt_builder", "generator")

experiment = Experiment(
    name="mmlu_gpt4_1_nano_no_perturbation",
    configs=pipeline.to_dict(),
    dataset="tasksource/mmlu_anatomy",
    model="gpt-4.1-nano-2025-04-14"
)

dataset_names = get_dataset_config_names("tasksource/mmlu")
predictions = defaultdict(list)
for dataset_name in tqdm(dataset_names, desc="Dataset"):
    data = load_dataset("tasksource/mmlu", name=dataset_name, split="test")
    for sample in tqdm(data, desc="Sample"):
        response = pipeline.run({
            "question": sample["question"],
            "choices": sample["choices"]
        })
        predictions["dataset"].append(dataset_name)
        predictions["question"].append(sample["question"])
        predictions["answer"].append(answer_mapping[sample["answer"]])
        predictions["prediction"].append(response["generator"]["replies"][0][0])
        predictions["output"].append(response["generator"]["replies"][0])


experiment.add_predictions(predictions)

# Calculate Exact Match
exact_match = exact_match(predictions["prediction"], predictions["answer"])
experiment.add_metrics({
    "exact_match": exact_match,
})

experiment.save("data/experiments/mmlu/no_perturbation.json")
