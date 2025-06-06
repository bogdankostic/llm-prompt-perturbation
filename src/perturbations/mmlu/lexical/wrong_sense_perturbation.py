from collections import defaultdict
from datetime import datetime
import os
import subprocess
import json

import pandas as pd
from datasets import load_dataset, get_dataset_config_names
from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.utils import Secret
from tqdm import tqdm

from src.custom_components.lexical.wrong_sense_variation import LexicalVariator
from src.custom_components.generator.cached_chat_generator import CachedOpenAIChatGenerator


answer_mapping = ["A", "B", "C", "D"]


# Construct perturbation pipeline
prompt_template = """Question: {{question}}
{% for choice in choices %}{{ ['A', 'B', 'C', 'D'][loop.index0] }}) {{choice}}
{% endfor %}
Answer:
"""

wsd_model = CachedOpenAIChatGenerator(
    api_key=Secret.from_env_var("PLACEHOLDER"),
    model="swap-uniba/LLM-wsd-FT-ALL",
    cache_dir="/experiments/llm-cache",
    api_base_url=f"{os.environ['WSD_MODEL_ENDPOINT']}/v1",
    generation_kwargs={
        "temperature": 0,
        "seed": 77
    }
)

lexical_variator = LexicalVariator(
    wsd_model=wsd_model,
    spacy_model="en_core_web_sm",
    wordnet_version="oewn:2024",
    random_seed=77
)
prompt_builder = PromptBuilder(
    template=prompt_template
)

perturbation_pipeline = Pipeline()
perturbation_pipeline.add_component(name="prompt_builder", instance=prompt_builder)
perturbation_pipeline.add_component(name="lexical_variator", instance=lexical_variator)
perturbation_pipeline.connect("prompt_builder.prompt", "lexical_variator.context")

# Perturb dataset
perturbed_data = defaultdict(list)
dataset_names = get_dataset_config_names("tasksource/mmlu")
for dataset_name in tqdm(dataset_names, desc="Dataset"):
    data = load_dataset("tasksource/mmlu", name=dataset_name, split="test")
    for sample in tqdm(data, desc="Sample"):
        # Perturb the question
        response = perturbation_pipeline.run({
            "prompt_builder": {
                "question": sample["question"],
                "choices": sample["choices"]
            },
            "lexical_variator": {"text": sample["question"]}
        })
        perturbed_question = response["lexical_variator"]["text"]
        question_metadata = response["lexical_variator"]["metadata"]

        # Perturb the answer choices
        perturbed_choices = []
        choices_metadata = []
        for idx, choice in enumerate(sample["choices"]):
            response = perturbation_pipeline.run({
                "prompt_builder": {
                    "question": sample["question"],
                    "choices": sample["choices"]
                },
                "lexical_variator": {"text": choice}
            })
            perturbed_choices.append(response["lexical_variator"]["text"])
            choices_metadata.append(response["lexical_variator"]["metadata"])

        perturbed_data["dataset"].append(dataset_name)
        perturbed_data["question"].append(perturbed_question)
        perturbed_data["answer"].append(answer_mapping[sample["answer"]])
        perturbed_data["choices"].append(perturbed_choices)
        perturbed_data["question_metadata"].append(question_metadata)
        perturbed_data["choice_metadata"].append(choices_metadata)

# Save perturbed data
perturbed_data_df = pd.DataFrame(perturbed_data)
output = {
    "metadata": {
        "description": f"MMLU dataset perturbed with wrong sense variations based on WordNet and Word Sense Disambiguation",
        "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "data": perturbed_data_df.to_dict(orient="records")
}

output_path = "/experiments/data/mmlu/lexical/wrong_sense_perturbation.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output, f)
