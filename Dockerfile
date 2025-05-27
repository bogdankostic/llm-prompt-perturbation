FROM python:3.12-slim

WORKDIR /llm_prompt_perturbation

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy only the necessary files first to leverage Docker cache
COPY pyproject.toml .
COPY .git/ .git/
COPY src/ src/

# Install dependencies
RUN pip install --no-cache-dir -e .
