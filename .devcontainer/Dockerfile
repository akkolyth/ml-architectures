FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN pip install --no-cache-dir \
    "numpy>=1.26.0" \
    "pandas>=2.1.0" \
    "scikit-learn>=1.3.0" \
    "matplotlib>=3.8.0" \
    "seaborn>=0.13.0" \
    "jupyter>=1.0.0" \
    "ipykernel>=6.25.0" \
    "ruff>=0.1.0" \
    "pytest>=7.4.0" \
    "pytest-cov>=4.1.0" \
    "torch>=2.1.0" \
    "torchvision>=0.16.0" \
    "tensorboard>=2.15.0" \
    "spacy" \
    "rapidfuzz" \
    "transformers" \
    "accelerate" \
    "tqdm" \
    "openai" \
    "sentence-transformers" \
    "metaphone" \
    && python -m spacy download pl_core_news_sm

WORKDIR /workspace

USER $USERNAME

ENV PATH="/home/$USERNAME/.local/bin:$PATH"

SHELL ["/bin/bash", "-c"]
