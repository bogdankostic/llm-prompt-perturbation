from collections import defaultdict
from datetime import datetime
import os
import subprocess
import json

import pandas as pd
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

# Clone the AMEGA dataset
subprocess.run(["git", "clone", "https://github.com/DATEXIS/AMEGA-benchmark.git"])

# Perturb cases
cases_df = pd.read_csv("AMEGA-benchmark/data/cases.csv", sep=";")
perturbed_cases = []
all_case_metadata = []
for case_str in tqdm(cases_df["case_str"], desc="Case"):
    response = perturbation_pipeline.run({"syntactic_variator": {"text": case_str}})
    
    perturbed_case_str = response["syntactic_variator"]["text"]
    case_metadata = response["syntactic_variator"]["metadata"]
    perturbed_cases.append(perturbed_case_str)
    all_case_metadata.append(case_metadata)

# Save perturbed cases
cases_df["case_str"] = perturbed_cases
cases_df["metadata"] = all_case_metadata
output = {
    "metadata": {
        "description": f"AMEGA cases perturbed with syntactic transformations",
        "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "data": cases_df.to_dict(orient="records")
}
output_path = "/experiments/data/amega/syntactic/syntactic_perturbation_cases.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output, f)

# Perturb questions
questions_df = pd.read_csv("AMEGA-benchmark/data/questions.csv", sep=";")
perturbed_questions = []
all_question_metadata = []
for question_str in tqdm(questions_df["question_str"], desc="Question"):
    response = perturbation_pipeline.run({"syntactic_variator": {"text": question_str}})
    
    perturbed_question_str = response["syntactic_variator"]["text"]
    question_metadata = response["syntactic_variator"]["metadata"]
    perturbed_questions.append(perturbed_question_str)
    all_question_metadata.append(question_metadata)

# Save perturbed questions
questions_df["question_str"] = perturbed_questions
questions_df["metadata"] = all_question_metadata
output = {
    "metadata": {
        "description": f"AMEGA questions perturbed with syntactic transformations",
        "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "data": questions_df.to_dict(orient="records")
}
output_path = "/experiments/data/amega/syntactic/syntactic_perturbation_questions.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output, f)
