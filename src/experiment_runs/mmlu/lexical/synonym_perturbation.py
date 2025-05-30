from collections import defaultdict
import os

from datasets import load_dataset, get_dataset_config_names
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
    api_base_url=f"{os.environ['WSD_MODEL_ENDPOINT']}/v1",
    generation_kwargs = {
        "temperature": 0,
        "seed": 77
    }
)

lexical_variator = LexicalVariator(
    wsd_model=wsd_model,
    random_seed=77
)

# Same instance cannot be used in multiple pipelines
prompt_builder_1 = PromptBuilder(
    template=prompt_template
)
prompt_builder_2 = PromptBuilder(
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

variation_pipeline = Pipeline()
variation_pipeline.add_component(name="prompt_builder", instance=prompt_builder_1)
variation_pipeline.add_component(name="lexical_variator", instance=lexical_variator)
variation_pipeline.connect("prompt_builder.prompt", "lexical_variator.context")

llm_pipeline = Pipeline()
llm_pipeline.add_component(name="prompt_builder", instance=prompt_builder_2)
llm_pipeline.add_component(name="generator", instance=generator)
llm_pipeline.connect("prompt_builder", "generator")

experiment = Experiment(
    name="mmlu_gpt4_1_nano_lexical_perturbation",
    configs={"lexical_variator": lexical_variator.to_dict(), "pipeline": llm_pipeline.to_dict()},
    dataset="tasksource/mmlu_anatomy",
    model="gpt-4.1-nano-2025-04-14"
)

dataset_names = get_dataset_config_names("tasksource/mmlu")
predictions = defaultdict(list)
for dataset_name in tqdm(dataset_names, desc="Dataset"):
    data = load_dataset("tasksource/mmlu", name=dataset_name, split="test")
    for sample in tqdm(data, desc="Sample"):
        # Preprocess: vary the question and choices separately
        variator_response = variation_pipeline.run({
            "prompt_builder": {
                "question": sample["question"],
                "choices": sample["choices"]
            },
            "lexical_variator": {"text": sample["question"]}
        })
        variated_question = variator_response["lexical_variator"]["text"]
        question_metadata = variator_response["lexical_variator"]["metadata"]
        
        variated_choices = []
        choices_metadata = []
        for idx, choice in enumerate(sample["choices"]):
            variator_response = variation_pipeline.run({
                "prompt_builder": {
                    "question": choice,
                    "choices": sample["choices"]
                },
                "lexical_variator": {"text": choice}
            })
            variated_choice = variator_response["lexical_variator"]["text"]
            choice_metadata = variator_response["lexical_variator"]["metadata"]
            variated_choices.append(variated_choice)
            choices_metadata.append(choice_metadata)

        response = llm_pipeline.run({
            "question": variated_question,
            "choices": variated_choices
        })
        predictions["dataset"].append(dataset_name)
        predictions["question"].append(sample["question"])
        predictions["answer"].append(answer_mapping[sample["answer"]])
        predictions["prediction"].append(response["generator"]["replies"][0][0])
        predictions["output"].append(response["generator"]["replies"][0])
        predictions["variated_question"].append(variated_question)
        predictions["variated_choices"].append(variated_choices)
        predictions["question_metadata"].append(question_metadata)
        predictions["choice_metadata"].append(choice_metadata)

experiment.add_predictions(predictions)

# Calculate Exact Match
exact_match = exact_match(predictions["prediction"], predictions["answer"])

# Add metrics to experiment
experiment.add_metrics({
    "exact_match": exact_match,
})

experiment.save("/experiments/mmlu/lexical/synonym_perturbation.json")
