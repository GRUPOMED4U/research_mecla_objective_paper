# Docker container to setup the environment to run the NER training and evaluation
FROM python:3.12-slim

# Install git and clean up apt lists to minimize image size
# The commands are combined into a single RUN instruction to create fewer Docker layers
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    rm -rf /var/lib/apt/lists/*

# Useful tools
RUN apt-get update && apt-get install -y vim procps

# Setup Google Cloud Storage
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg curl

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

RUN apt-get update && apt-get install -y google-cloud-cli

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir --upgrade uv

RUN git clone https://almeidava93-spesia:${GITHUB_TOKEN}@github.com/GRUPOMED4U/research_mecla_objective_paper.git

WORKDIR /app/research_mecla_objective_paper

ENV GOOGLE_APPLICATION_CREDENTIALS=/app/key.json
ENV UV_HTTP_TIMEOUT=600
ENV UV_HTTP_RETRIES=5

RUN uv sync

RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

RUN chmod +x ./docker/entrypoint.sh

ENTRYPOINT ["./docker/entrypoint.sh"]

CMD ["sleep", "infinity"]