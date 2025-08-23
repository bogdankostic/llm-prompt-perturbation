from collections import defaultdict
from datetime import datetime
import os
import subprocess
import json

import pandas as pd
from datasets import load_dataset
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
    api_base_url=f"{os.environ.get('LLM_MODEL_ENDPOINT', 'http://localhost:8000')}/v1",
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
data = load_dataset("rajpurkar/squad", split="validation").shuffle(seed=77).select(range(1000))
for sample in tqdm(data, desc="Sample"):
    question_response = perturbation_pipeline.run({"syntactic_variator": {"text": sample["question"]}})
    context_response = perturbation_pipeline.run({"syntactic_variator": {"text": sample["context"]}})

    question = question_response["syntactic_variator"]["text"]
    question_metadata = question_response["syntactic_variator"]["metadata"]
    context = context_response["syntactic_variator"]["text"]
    context_metadata = context_response["syntactic_variator"]["metadata"]
    perturbed_data["question"].append(question)
    perturbed_data["context"].append(context)
    perturbed_data["question_metadata"].append(question_metadata)
    perturbed_data["context_metadata"].append(context_metadata)
    perturbed_data["id"].append(sample["id"])
    perturbed_data["answers"].append(sample["answers"])

# Save perturbed data
perturbed_data_df = pd.DataFrame(perturbed_data)
output = {
    "metadata": {
        "description": f"SQuAD dataset perturbed with syntactic transformations",
        "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "data": perturbed_data_df.to_dict(orient="records")
}

output_path = "/experiments/data/squad/syntactic/syntactic_perturbation.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output, f)
