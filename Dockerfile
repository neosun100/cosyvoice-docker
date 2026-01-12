# =============================================================================
# CosyVoice TTS Service - Production Dockerfile
# =============================================================================
# Features:
#   - Multi-stage build for optimized image size
#   - vLLM acceleration support (runtime configurable)
#   - Multi-model support via volume mount
#   - No embedded model download - models mounted at runtime
#
# Usage:
#   docker build -t cosyvoice:latest .
#   docker run -d --gpus all -p 8188:8188 \
#     -v /path/to/models:/models:ro \
#     -e MODEL_DIR=/models/Fun-CosyVoice3-0.5B \
#     cosyvoice:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base System
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        unzip \
        ffmpeg \
        sox \
        libsox-dev \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# -----------------------------------------------------------------------------
# Stage 2: Conda Environment
# -----------------------------------------------------------------------------
FROM base AS conda

# Install Miniforge (lightweight conda)
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

# Create Python environment with pynini (required for text normalization)
RUN conda create -n cosyvoice python=3.10 -y \
    && conda install -n cosyvoice -c conda-forge pynini==2.1.5 -y \
    && conda clean -afy

# -----------------------------------------------------------------------------
# Stage 3: Python Dependencies
# -----------------------------------------------------------------------------
FROM conda AS deps

ENV PATH=/opt/conda/envs/cosyvoice/bin:$PATH \
    CONDA_DEFAULT_ENV=cosyvoice

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
COPY backup/ttsfrd_dependency-0.1-py3-none-any.whl backup/ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl ./

# Install ttsfrd for better text normalization (optional)
RUN pip install --no-cache-dir ttsfrd_dependency-0.1-py3-none-any.whl \
    && pip install --no-cache-dir ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl \
    && rm -f *.whl

RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    && rm -rf ~/.cache/pip /tmp/*
    

# -----------------------------------------------------------------------------
# Stage 4: Application Runtime
# -----------------------------------------------------------------------------
FROM deps AS runtime

# Copy application code
COPY third_party/Matcha-TTS third_party/Matcha-TTS/
COPY cosyvoice cosyvoice/
COPY server.py voice_manager.py ./

# Copy and extract ttsfrd resource files for better text normalization
COPY backup/resource.zip /tmp/resource.zip
RUN mkdir -p /app/pretrained_models/CosyVoice-ttsfrd \
    && unzip -q /tmp/resource.zip -d /app/pretrained_models/CosyVoice-ttsfrd/ \
    && rm -f /tmp/resource.zip

# Python path configuration (required for imports)
ENV PYTHONPATH=/app:/app/third_party/Matcha-TTS

# Create mount point directories
RUN mkdir -p /models /data/output /data/voices /root/.cache

# Set shell for runtime
SHELL ["conda", "run", "--no-capture-output", "-n", "cosyvoice", "/bin/bash", "-c"]
