from collections import defaultdict
from datetime import datetime
import os
import subprocess
import json
from typing import List, Tuple
import uuid

import pandas as pd
from datasets import load_dataset
from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from pydantic import BaseModel
from tqdm import tqdm

from src.custom_components.generator.cached_chat_generator import CachedOpenAIChatGenerator


class QuestionParagraph(BaseModel):
    question: str
    paragraph: str
    changes: List[Tuple[str, str]]

json_schema = QuestionParagraph.model_json_schema()

# Construct perturbation pipeline
prompt_template = [ChatMessage.from_user("""You are given a question and a paragraph related to the question.
Create a version of the question and the paragraph where you exchange words by their synonyms, if possible. The rest should be kept the same, especially the meaning. The language should stay natural. The selected synonyms should respect the terminology of the question's domain. If a synonym is not possible, keep the original word.
You are also given a list of substrings that should not be changed.
Also output a list of tuples, where each tuple contains the original word and its corresponding new word. The list must include EVERY word in the original text in order, even if the same word appears multiple times. Include unchanged words as well as stop words and other non-content words, for example ["the", "the"] or ["for", "for"].
                                         
The output should be a valid JSON object with the following fields:
- question: str  # The question with synonyms
- paragraph: str  # The paragraph with synonyms
- changes: List[Tuple[str, str]]  # The list of tuples containing all original words and new words

Question: {{question}}
Paragraph: {{paragraph}}
Substrings to keep unchanged: {{substrings}}
""")]

prompt_builder = ChatPromptBuilder(
    template=prompt_template
)
paraphrase_model = CachedOpenAIChatGenerator(
    api_key=Secret.from_env_var("PLACEHOLDER"),
    model="google/gemma-3-27b-it",
    cache_dir="/experiments/llm-cache",
    api_base_url=f"{os.environ['LLM_MODEL_ENDPOINT']}/v1",
    timeout=300,
    generation_kwargs={
        "temperature": 0,
        "seed": 77,
        "extra_body": {
            "guided_json": json_schema,
            "request_id": f"squad-{uuid.uuid4()}"
        }
    }
)

perturbation_pipeline = Pipeline()
perturbation_pipeline.add_component(name="prompt_builder", instance=prompt_builder)
perturbation_pipeline.add_component(name="paraphrase_model", instance=paraphrase_model)
perturbation_pipeline.connect("prompt_builder.prompt", "paraphrase_model.messages")


# Perturb dataset
perturbed_data = defaultdict(list)
data = load_dataset("rajpurkar/squad", split="validation")
for sample in tqdm(data, desc="Sample"):
    answer_strings = list(set(sample["answers"]["text"]))
    response = perturbation_pipeline.run({
        "prompt_builder": {
            "question": sample["question"],
            "paragraph": sample["context"],
            "substrings": answer_strings
        }
    })

    try:
        response_dict = json.loads(response["paraphrase_model"]["replies"][0].text)
    except json.JSONDecodeError:
        print(f"Error parsing JSON for sample {sample['question']}")
        response_dict = {
            "question": sample["question"],
            "paragraph": sample["context"],
            "changes": []
        }

    question = response_dict["question"]
    paragraph = response_dict["paragraph"]
    changes = response_dict["changes"]
    perturbed_data["question"].append(question)
    perturbed_data["paragraph"].append(paragraph)
    perturbed_data["changes"].append(changes)

# Save perturbed data
perturbed_data_df = pd.DataFrame(perturbed_data)
output = {
    "metadata": {
        "description": f"SQuAD dataset perturbed with synonyms via LLM",
        "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "data": perturbed_data_df.to_dict(orient="records")
}

output_path = "/experiments/data/squad/lexical/llm_synonym_perturbation.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output, f)
