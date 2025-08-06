FROM python:3.12-slim

WORKDIR /llm_prompt_perturbation

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set-up SSH
RUN apt-get update && \
    apt-get -yq install openssh-server && \
    mkdir -p /var/run/sshd

# Copy necessary files
COPY pyproject.toml .
COPY .git/ .git/
COPY src/ src/

# Install dependencies
RUN pip install --no-cache-dir -e .
RUN python -m spacy download en_core_web_trf

# SSH Server
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]
