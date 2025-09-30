import os
import subprocess
import json
import pandas as pd
import vertexai
from google.oauth2 import service_account

from src.custom_components.evaluator.amega import AMEGAEvaluator


def main():
    # Initialize Vertex AI
    with open("/experiments/google/auth.json", "r") as f:
        config = json.load(f)
    credentials = service_account.Credentials.from_service_account_file("/experiments/google/auth.json")
    vertexai.init(project=config["project_id"], credentials=credentials, location="global")

    # Clone the AMEGA dataset
    subprocess.run(["git", "clone", "https://github.com/DATEXIS/AMEGA-benchmark.git"])
    # Load dataset
    criteria_df = pd.read_csv("AMEGA-benchmark/data/criteria.csv", sep=";")

    results_dir = "/experiments"
    # Initialize evaluator
    evaluator = AMEGAEvaluator(model="gemini-2.0-flash-lite-001")
    for item in os.listdir(results_dir):
        current_dir = os.path.join(results_dir, item)
        if os.path.isdir(current_dir) and item.startswith(("mistral", "qwen", "gemini", "gemma", "gpt", "llama")):
            # Original AMEGA set
            original_amega_file = os.path.join(current_dir, "amega/original.json")
            syntactic_variation_file = os.path.join(current_dir, "amega/syntactic/syntactic.json")
            lexical_variation_file = os.path.join(current_dir, "amega/lexical/llm_synonym.json")

            for file in [original_amega_file, syntactic_variation_file, lexical_variation_file]:
                new_majority_vote = []
                new_true_rates = []
                new_confidence_rate = []
                new_fail_rate = []
                with open(file, "r") as f:
                    data = json.load(f)
                for model_answer, case_id, question_id in zip(
                    data["predictions"]["model_answer"],
                    data["predictions"]["case_id"],
                    data["predictions"]["question_id"],
                ):
                    criteria_list = criteria_df.loc[
                        (criteria_df["case_id"] == case_id)
                        & (criteria_df["question_id"] == question_id),
                        "criteria_str",
                    ].values.tolist()

                    evaluator_response = evaluator.run(model_answer, criteria_list)
                    majority_vote = evaluator_response["majority_vote"]
                    true_rates = evaluator_response["true_rates"]
                    confidence_rate = evaluator_response["confidence_rate"]
                    fail_rate = evaluator_response["fail_rate"]

                    new_majority_vote.append(majority_vote)
                    new_true_rates.append(true_rates)
                    new_confidence_rate.append(confidence_rate)
                    new_fail_rate.append(fail_rate)

                data["predictions"]["majority_vote"] = new_majority_vote
                data["predictions"]["true_rates"] = new_true_rates
                data["predictions"]["confidence_rate"] = new_confidence_rate
                data["predictions"]["fail_rate"] = new_fail_rate

                new_file = file.replace(".json", "_v3.json")
                with open(new_file, "w") as f:
                    json.dump(data, f)

if __name__ == "__main__":
    main()