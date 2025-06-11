from collections import defaultdict
from datetime import datetime
import os
import subprocess
import json
import sys
from typing import List, Tuple

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
    changes: List[Tuple[str, str]]


json_schema = MultipleChoiceQuestion.model_json_schema()
answer_mapping = ["A", "B", "C", "D"]


# Construct perturbation pipeline
prompt_template = [ChatMessage.from_user("""You are given a multiple-choice question.
Create a version of the question and the answer choices where you exchange words by their synonyms, the rest should be kept the same, especially the meaning.
Output also a list of tuples containing the original word and the new word. The list should contain an entry for each word, so also for unchanged words.

The output should be a valid JSON object with the following fields:
- question: str  # The question with synonyms
- choices: List[str]  # The answer choices with synonyms
- changes: List[Tuple[str, str]]  # The list of tuples containing the original word and the new word

Question: {{question}}
{% for choice in choices %}{{ ['A', 'B', 'C', 'D'][loop.index0] }}) {{choice}}
{% endfor %}
""")]

prompt_builder = ChatPromptBuilder(
    template=prompt_template
)
paraphrase_model = CachedOpenAIChatGenerator(
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    #model="meta-llama/Llama-3.1-8B-Instruct",
    model="gpt-4.1-nano-2025-04-14",
    cache_dir="tmp/llm-cache",
    #api_base_url=f"{os.environ['LLM_MODEL_ENDPOINT']}/v1",
    #api_base_url="http://localhost:8000/v1",
    generation_kwargs={
        "temperature": 0,
        "seed": 77,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "MultipleChoiceQuestion",
                "schema": json_schema
            }
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
                "choices": sample["choices"],
                "changes": []
            }

        question = response_dict["question"]
        choices = response_dict["choices"]
        changes = response_dict["changes"]
        for idx, choice in enumerate(choices):
            if choice[:2] in ["A)", "B)", "C)", "D)"]:
                choices[idx] = choice[2:].strip()
        answer = answer_mapping[sample["answer"]]
        perturbed_data["question"].append(question)
        perturbed_data["choices"].append(choices)
        perturbed_data["changes"].append(changes)
        perturbed_data["answer"].append(answer)
        perturbed_data["dataset_name"].append(dataset_name)

# Save perturbed data
perturbed_data_df = pd.DataFrame(perturbed_data)
output = {
    "metadata": {
        "description": f"MMLU dataset perturbed with synonyms via LLM",
        "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "data": perturbed_data_df.to_dict(orient="records")
}

output_path = "/experiments/data/mmlu/lexical/llm_synonym_perturbation.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output, f)
