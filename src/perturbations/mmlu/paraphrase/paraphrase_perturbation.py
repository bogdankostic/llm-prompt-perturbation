from collections import defaultdict
from datetime import datetime
import os
import subprocess
import json
from typing import List

import pandas as pd
from datasets import load_dataset, get_dataset_config_names
from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from pydantic import BaseModel
from tqdm import tqdm

from src.custom_components.generator.cached_chat_generator import CachedOpenAIChatGenerator


class MultipleChoiceQuestion(BaseModel):
    question: str
    choices: List[str]


json_schema = MultipleChoiceQuestion.model_json_schema()
answer_mapping = ["A", "B", "C", "D"]


# Construct perturbation pipeline
prompt_template = [ChatMessage.from_user("""You are given a multiple-choice question. 
Your task is to rewrite the question and its answer choices using different wording and syntactic structures, but without changing the original meaning. 
Ensure the paraphrased version remains clear and preserves the structure of a multiple-choice question with four answer options.
IMPORTANT: Do not add or remove any information.
The output should be a valid JSON object with the following fields:
- question: str  # The paraphrased question containing all relevant information
- choices: List[str]  # The paraphrased answer choices

Question: {{question}}
{% for choice in choices %}{{ ['A', 'B', 'C', 'D'][loop.index0] }}) {{choice}}
{% endfor %}
Answer:
""")]

prompt_builder = ChatPromptBuilder(
    template=prompt_template
)
paraphrase_model = CachedOpenAIChatGenerator(
    api_key=Secret.from_env_var("PLACEHOLDER"),
    model="meta-llama/Llama-3.1-8B-Instruct",
    cache_dir="/experiments/llm-cache",
    api_base_url=f"{os.environ['LLM_MODEL_ENDPOINT']}/v1",
    generation_kwargs={
        "temperature": 0,
        "seed": 77,
        "extra_body": {
            "guided_json": json_schema
        }
    }
)

perturbation_pipeline = Pipeline()
perturbation_pipeline.add_component(name="prompt_builder", instance=prompt_builder)
perturbation_pipeline.add_component(name="paraphrase_model", instance=paraphrase_model)
perturbation_pipeline.connect("prompt_builder.prompt", "paraphrase_model.messages")


# Perturb dataset
perturbed_data = defaultdict(list)
dataset_names = get_dataset_config_names("tasksource/mmlu")
for dataset_name in tqdm(dataset_names, desc="Dataset"):
    print(f"Processing dataset: {dataset_name}")
    data = load_dataset("tasksource/mmlu", name=dataset_name, split="test")
    for sample in tqdm(data, desc="Sample"):
        response = perturbation_pipeline.run({
            "prompt_builder": {
                "question": sample["question"],
                "choices": sample["choices"]
            }
        })

        try:
            response_dict = json.loads(response["paraphrase_model"]["replies"][0].text)
        except json.JSONDecodeError:
            print(f"Error parsing JSON for sample {sample['question']}")
            response_dict = {
                "question": sample["question"],
                "choices": sample["choices"]
            }
        question = response_dict["question"]
        choices = response_dict["choices"]
        for idx, choice in enumerate(choices):
            if choice[:2] in ["A)", "B)", "C)", "D)"]:
                choices[idx] = choice[2:].strip()
        answer = answer_mapping[sample["answer"]]
        perturbed_data["question"].append(question)
        perturbed_data["choices"].append(choices)
        perturbed_data["answer"].append(answer)
        perturbed_data["dataset"].append(dataset_name)

# Save perturbed data
perturbed_data_df = pd.DataFrame(perturbed_data)
output = {
    "metadata": {
        "description": f"MMLU dataset perturbed with paraphrased questions and answer choices",
        "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "data": perturbed_data_df.to_dict(orient="records")
}

output_path = "/experiments/data/mmlu/paraphrase/paraphrase_perturbation.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output, f)
