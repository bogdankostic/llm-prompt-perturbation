from collections import defaultdict

from datasets import load_dataset, get_dataset_config_names
from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from tqdm import tqdm

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
prompt_builder = ChatPromptBuilder(
    template=chat_messages
)
generator = CachedOpenAIChatGenerator(
    api_base_url="http://meta-llama-llama-3-2-1b-instruct.bdarabisahneh.svc.cluster.local:8000/v1",
    cache_dir="/experiments/llm-cache",
    model="meta-llama/Llama-3.2-1B-Instruct",
    api_key=Secret.from_env_var("PLACEHOLDER"),
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
    name="mmlu_llama_3_2_1b_no_perturbation",
    configs=pipeline.to_dict(),
    dataset="tasksource/mmlu",
    model="meta-llama/Llama-3.2-1B-Instruct "
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

experiment.save("/experiments/mmlu/no_perturbation_llama_3-2-1b.json")
