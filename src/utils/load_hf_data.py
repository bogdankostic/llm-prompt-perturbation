from typing import Dict, List, Optional
import json
import os
import subprocess
from datetime import datetime
from collections import defaultdict

from datasets import load_dataset, get_dataset_config_names
import pandas as pd
from tqdm import tqdm


def load_mmlu_data(
    dataset_name: Optional[str] = None,
    split: str = "test",
    return_as_dataframe: bool = True
) -> Dict[str, List] | pd.DataFrame:
    """
    Load MMLU dataset from HuggingFace.
    
    Args:
        dataset_name: Specific MMLU dataset name. If None, loads all datasets.
        split: Dataset split to load (default: "test")
        return_as_dataframe: Whether to return data as pandas DataFrame (default: True)
    
    Returns:
        Dictionary with lists of data or pandas DataFrame containing:
        - dataset: Name of the dataset
        - question: Question text
        - answer: Answer letter (A, B, C, D)
        - choices: List of answer choices
    """
    data = defaultdict(list)
    
    answer_mapping = ["A", "B", "C", "D"]
    
    if dataset_name is None:
        dataset_names = get_dataset_config_names("tasksource/mmlu")
    else:
        dataset_names = [dataset_name]
        
    for ds_name in tqdm(dataset_names, desc="Loading datasets"):
        dataset = load_dataset("tasksource/mmlu", name=ds_name, split=split)
        
        for sample in tqdm(dataset, desc=f"Processing {ds_name}"):
            data["dataset"].append(ds_name)
            data["question"].append(sample["question"])
            data["answer"].append(answer_mapping[sample["answer"]])
            data["choices"].append(sample["choices"])
    
    if return_as_dataframe:
        return pd.DataFrame(data)
    return data


def save_mmlu_data(data: pd.DataFrame, output_path: str):
    """
    Save MMLU data in JSON format with metadata.
    
    Args:
        data: DataFrame containing the MMLU data
        output_path: Path where to save the JSON file
    """
    output = {
        "metadata": {
            "description": "MMLU dataset loaded from HuggingFace",
            "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip(),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "data": data.to_dict(orient="records")
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    # Load all datasets
    all_data = load_mmlu_data()
    print(f"Loaded {len(all_data)} samples from all datasets")
    
    # Save to experiments directory
    output_path = "/experiments/data/mmlu/original.json"
    save_mmlu_data(all_data, output_path)
