from collections import defaultdict
from datetime import datetime
import os
import subprocess
import json
from typing import List, Tuple

import pandas as pd
from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from pydantic import BaseModel
from tqdm import tqdm

from src.custom_components.generator.cached_chat_generator import CachedOpenAIChatGenerator


# Construct perturbation pipeline for cases
class ClinicalCase(BaseModel):
    text: str
    changes: List[Tuple[str, str]]


json_schema = ClinicalCase.model_json_schema()

prompt_template = [ChatMessage.from_user("""You are given a clinical case.
Create a version of the clinical case where you exchange words by their synonyms, if possible. The rest should be kept the same, especially the meaning. The language should stay natural. The selected synonyms should respect the terminology of the clinical case's domain. If a synonym is not possible, keep the original word.
Also output a list of tuples, where each tuple contains the original word and its corresponding new word. The list must include EVERY word in the original text in order, even if the same word appears multiple times. Include unchanged words as well as stop words and other non-content words, for example ["the", "the"] or ["for", "for"].
                                         
The output should be a valid JSON object with the following fields:
- text: str  # The clinical case with synonyms
- changes: List[Tuple[str, str]]  # The list of tuples containing all original words and new words

Clinical Case: {{text}}
""")]

prompt_builder = ChatPromptBuilder(
    template=prompt_template
)
paraphrase_model = CachedOpenAIChatGenerator(
    api_key=Secret.from_env_var("PLACEHOLDER"),
    model="RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8",
    cache_dir="/experiments/llm-cache",
    api_base_url=f"{os.environ['LLM_MODEL_ENDPOINT']}/v1",
    timeout=600,
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

# Clone the AMEGA dataset
subprocess.run(["git", "clone", "https://github.com/DATEXIS/AMEGA-benchmark.git"])

# Perturb cases
cases_df = pd.read_csv("AMEGA-benchmark/data/cases.csv", sep=";")
perturbed_cases = []
all_changes = []
for case_str in tqdm(cases_df["case_str"], desc="Case"):
    response = perturbation_pipeline.run({
        "prompt_builder": {
            "text": case_str
        }
    })

    try:
        response_dict = json.loads(response["paraphrase_model"]["replies"][0].text)
    except json.JSONDecodeError:
        print(f"Error parsing JSON for case {case_str}")
        response_dict = {
            "text": case_str,
            "changes": []
        }

    perturbed_case_str = response_dict["text"]
    case_changes = response_dict["changes"]
    perturbed_cases.append(perturbed_case_str)
    all_changes.append(case_changes)

# Save perturbed cases
cases_df["case_str"] = perturbed_cases
cases_df["changes"] = all_changes
output = {
    "metadata": {
        "description": f"AMEGA cases perturbed with synonyms via LLM",
        "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "data": cases_df.to_dict(orient="records")
}
output_path = "/experiments/data/amega/lexical/llm_synonym_perturbation_cases.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output, f)


# Construct perturbation pipeline for questions
class ClinicalQuestion(BaseModel):
    question: str
    changes: List[Tuple[str, str]]


json_schema = ClinicalQuestion.model_json_schema()

prompt_template = [ChatMessage.from_user("""You are given a clinical question.
Create a version of the clinical question where you exchange words by their synonyms, if possible. The rest should be kept the same, especially the meaning. The language should stay natural. The selected synonyms should respect the terminology of the clinical question's domain. If a synonym is not possible, keep the original word.
Also output a list of tuples, where each tuple contains the original word and its corresponding new word. The list must include EVERY word in the original text in order, even if the same word appears multiple times. Include unchanged words as well as stop words and other non-content words, for example ["the", "the"] or ["for", "for"].
                                         
The output should be a valid JSON object with the following fields:
- question: str  # The clinical question with synonyms
- changes: List[Tuple[str, str]]  # The list of tuples containing all original words and new words

Clinical Question: {{question}}
""")]

prompt_builder = ChatPromptBuilder(
    template=prompt_template
)
paraphrase_model = CachedOpenAIChatGenerator(
    api_key=Secret.from_env_var("PLACEHOLDER"),
    model="RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8",
    cache_dir="/experiments/llm-cache",
    api_base_url=f"{os.environ['LLM_MODEL_ENDPOINT']}/v1",
    timeout=600,
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


# Perturb questions
questions_df = pd.read_csv("AMEGA-benchmark/data/questions.csv", sep=";")
perturbed_questions = []
all_changes = []
for question_str in tqdm(questions_df["question_str"], desc="Question"):
    response = perturbation_pipeline.run({
        "prompt_builder": {
            "question": question_str
        }
    })

    try:
        response_dict = json.loads(response["paraphrase_model"]["replies"][0].text)
    except json.JSONDecodeError:
        print(f"Error parsing JSON for question {question_str}")
        response_dict = {
            "question": question_str,
            "changes": []
        }

    perturbed_question_str = response_dict["question"]
    question_changes = response_dict["changes"]
    perturbed_questions.append(perturbed_question_str)
    all_changes.append(question_changes)

# Save perturbed questions
questions_df["question_str"] = perturbed_questions
questions_df["changes"] = all_changes
output = {
    "metadata": {
        "description": f"AMEGA questions perturbed with synonyms via LLM",
        "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "data": questions_df.to_dict(orient="records")
}
output_path = "/experiments/data/amega/lexical/llm_synonym_perturbation_questions.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output, f)
