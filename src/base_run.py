from datasets import load_dataset
from haystack import Pipeline
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.utils import Secret
from tqdm import tqdm
import pandas as pd


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
    api_key=Secret.from_token("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"),
    generation_kwargs={
        "temperature": 0,
        "seed": 77,
    }
)

pipeline = Pipeline()
pipeline.add_component(name="prompt_builder", instance=prompt_builder)
pipeline.add_component(name="generator", instance=generator)
pipeline.connect("prompt_builder", "generator")

ds = load_dataset("tasksource/mmlu", "anatomy", split="test")
predictions = []
for sample in tqdm(ds):
    response = pipeline.run({
        "question": sample["question"],
        "choices": sample["choices"]
    })
    predictions.append({
        "question": sample["question"],
        "answer": answer_mapping[sample["answer"]],
        # Use only the first letter of the generated output
        "prediction": response["generator"]["replies"][0][0],
        "output": response["generator"]["replies"][0]
    })

predictions = pd.DataFrame(predictions)
predictions.to_csv("data/predictions.csv", index=False)

