FROM python:3.12-slim

WORKDIR /llm_prompt_perturbation

# Copy only the necessary files first to leverage Docker cache
COPY pyproject.toml .
COPY .git/ .git/
COPY src/ src/

# Install dependencies
RUN pip install --no-cache-dir -e .
