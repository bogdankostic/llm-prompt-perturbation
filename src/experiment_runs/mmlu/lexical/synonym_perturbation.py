from collections import defaultdict

from datasets import load_dataset
from haystack import Pipeline
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.utils import Secret
from tqdm import tqdm

from src.utils.experiment import Experiment
from src.utils.metrics import exact_match
from src.custom_components.lexical.synonym_variation import LexicalVariator


answer_mapping = ["A", "B", "C", "D"]


prompt_template = """Question: {{question}}
{% for choice in choices %}{{ ['A', 'B', 'C', 'D'][loop.index0] }}) {{choice}}
{% endfor %}
Answer:
"""
system_prompt = """You are a helpful assistant that can answer multiple choice questions about the world. Answer the question based on the choices provided. Only output the letter of the choice that you think is the correct answer."""

# Create lexical variator for preprocessing
wsd_model = OpenAIChatGenerator(
    api_key=Secret.from_env_var("PLACEHOLDER"),
    model="swap-uniba/LLM-wsd-FT-ALL",
    api_base_url="http://localhost:8000/v1",
    generation_kwargs = {"max_tokens": 512}
)

lexical_variator = LexicalVariator(
    wsd_model=wsd_model,
    random_seed=77
)

prompt_builder = PromptBuilder(
    template=prompt_template
)
generator = OpenAIGenerator(
    model="gpt-4.1-nano-2025-04-14",
    system_prompt=system_prompt,
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
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
    name="mmlu_gpt4_1_nano_lexical_perturbation",
    configs={"lexical_variator": lexical_variator.to_dict(), "pipeline": pipeline.to_dict()},
    dataset="tasksource/mmlu_anatomy",
    model="gpt-4.1-nano-2025-04-14"
)

#ds = load_dataset("tasksource/mmlu", "anatomy", split="test")
ds = load_dataset("tasksource/mmlu", split="train")
predictions = defaultdict(list)
for sample in tqdm(ds):
    # Preprocess: vary the question and choices separately
    variated_question = lexical_variator.run(
        sample["question"], 
        additional_context="\n".join(sample["choices"])
    )["text"]
    variated_choices = []
    for idx, choice in enumerate(sample["choices"]):
        variated_choice = lexical_variator.run(
            choice, 
            additional_context=sample["question"] + f"\n{'\n'.join([sample['choices'][i] for i in range(len(sample['choices'])) if i != idx])}"
        )["text"]
        variated_choices.append(variated_choice)

    response = pipeline.run({
        "question": variated_question,
        "choices": variated_choices
    })
    predictions["question"].append(sample["question"])
    predictions["answer"].append(answer_mapping[sample["answer"]])
    predictions["prediction"].append(response["generator"]["replies"][0][0])
    predictions["output"].append(response["generator"]["replies"][0])
    predictions["variated_question"].append(variated_question)
    predictions["variated_choices"].append(variated_choices)


experiment.add_predictions(predictions)

# Calculate Exact Match
exact_match = exact_match(predictions["prediction"], predictions["answer"])

# Add metrics to experiment
experiment.add_metrics({
    "exact_match": exact_match,
})

experiment.save("data/experiments/mmlu/lexical/synonym_perturbation.json")
