# Performance Variance of Large Language Models under Prompt Perturbtion

This repository accompanies the master's thesis "*Performance Variance of Large Language Models under Prompt Perturbtion*", which studies how meaning-preserving lexical and syntactic changes influence the performance of LLMs on benchmark tasks.
Experiments cover the MMLU, SQuAD and AMEGA benchmarks. It contains the code needed to reprodcue the perturbation pipelines, evaluation jobs and analysis described in the thesis.

# Repository Structure
| Path                 | Description                                                                                                                                                                                                                                                                                                                                                     |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/`               | Source code packaged as a Python module.  It contains the pipelines for generating lexical and syntactic perturbations, wrappers around LLM APIs, utilities and custom components such as the `CachedOpenAIChatGenerator`.
| `k8s/perturbations/` | Kubernetes job definitions for generating perturbed benchmark datasets.  YAML files are organised by benchmark (`mmlu`, `squad`, `amega`) and perturbation type (`lexical` vs. `syntactic`).
| `k8s/inferences/`    | Kubernetes job definitions for running inference on the perturbed and original datasets.  Jobs are grouped by model name and dataset. 
| `src/notebooks/`     | Jupyter notebooks used for post‑processing and analysis of results.  These notebooks compute different statistics and generate tables and figures.                     |
| `pyproject.toml`     | Defines project dependencies. Installing the package with `pip install -e .` installs the required libraries.                                                                                                                                                                                           |
| `Dockerfile`         | Container definition for running the experiments.             |
| `data/`         | Contains perturbed versions of MMLU, SQuAD, and AMEGA.      |
| `results/`         | Contains the evaluation results for each model.      |
