from collections import defaultdict
from datetime import datetime
import os
import subprocess
import json

import pandas as pd
from datasets import load_dataset, get_dataset_config_names
from haystack import Pipeline
from haystack.utils import Secret
from tqdm import tqdm

from src.custom_components.generator.cached_chat_generator import CachedOpenAIChatGenerator
from src.custom_components.syntactic.syntactic_variation import SyntacticVariator


# Initialize components
transformation_model = CachedOpenAIChatGenerator(
    api_key=Secret.from_env_var("PLACEHOLDER"),
    model="RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8",
    cache_dir="/experiments/llm-cache",
    api_base_url=f"{os.environ['LLM_MODEL_ENDPOINT']}/v1",
    generation_kwargs={
        "temperature": 0,
        "seed": 77,
    }
)
syntactic_variator = SyntacticVariator(
    transformation_model=transformation_model,
    random_seed=77,
)

# Initialize pipeline
perturbation_pipeline = Pipeline()
perturbation_pipeline.add_component(name="syntactic_variator", instance=syntactic_variator)

# Perturb dataset
perturbed_data = defaultdict(list)
dataset_names = get_dataset_config_names("tasksource/mmlu")
for dataset_name in tqdm(dataset_names, desc="Dataset"):
    print(f"Processing dataset: {dataset_name}")
    data = load_dataset("tasksource/mmlu", name=dataset_name, split="test")
    for sample in tqdm(data, desc="Sample"):
        question_response = perturbation_pipeline.run({"syntactic_variator": {"text": sample["question"]}})
        choice_responses = []
        for choice in sample["choices"]:
            choice_response = perturbation_pipeline.run({"syntactic_variator": {"text": choice}})
            choice_responses.append(choice_response)

        question = question_response["syntactic_variator"]["text"]
        question_metadata = question_response["syntactic_variator"]["metadata"]
        choices = [choice_response["syntactic_variator"]["text"] for choice_response in choice_responses]
        choices_metadata = [choice_response["syntactic_variator"]["metadata"] for choice_response in choice_responses]
        perturbed_data["question"].append(question)
        perturbed_data["choices"].append(choices)
        perturbed_data["question_metadata"].append(question_metadata)
        perturbed_data["choices_metadata"].append(choices_metadata)
        perturbed_data["answer"].append(sample["answer"])
        perturbed_data["dataset"].append(dataset_name)

# Save perturbed data
perturbed_data_df = pd.DataFrame(perturbed_data)
output = {
    "metadata": {
        "description": f"MMLU dataset perturbed with syntactic transformations",
        "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "data": perturbed_data_df.to_dict(orient="records")
}

output_path = "/experiments/data/mmlu/syntactic/syntactic_perturbation.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output, f)
