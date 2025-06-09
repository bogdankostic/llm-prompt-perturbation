import json
import os
from typing import Any, Dict, List
from datetime import datetime
import subprocess


class Experiment:
    """
    Class for tracking experiment results.
    """

    def __init__(
        self,
        name: str,
        configs: Dict[str, Any],
        dataset: str,
        model: str,
        perturbation_type: str,
    ):
        self.name = name
        self.configs = configs
        self.dataset = dataset
        self.model = model
        self.perturbation_type = perturbation_type
        self.time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        self.predictions = {}
        self.metrics = {}

    def add_predictions(self, predictions: Dict[str, List]):
        self.predictions = predictions

    def add_metrics(self, metrics: Dict[str, Any]):
        self.metrics = metrics

    def save(self, path: str):
        output = {
            "name": self.name,
            "dataset": self.dataset,
            "model": self.model,
            "perturbation_type": self.perturbation_type,
            "configs": self.configs,
            "time": self.time,
            "commit_hash": self.commit_hash,
            "metrics": self.metrics ,
            "predictions": self.predictions,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f)
