# ── ValonyLabs Studio v3.0 ─────────────────────────────────────
# Multi-stage build: Node (frontend) → Python (backend + serve)
#
# Build targets:
#   default  — CUDA-enabled (g5.xlarge / L4 / A100)
#   --target cpu  — CPU-only (Fargate / local dev / CI)
#
# Usage:
#   docker build -t valonylabs-studio .
#   docker build -t valonylabs-studio:cpu --target cpu .
#   docker compose up                    # local dev with GPU
#   docker compose -f docker-compose.yml -f docker-compose.cpu.yml up
# ───────────────────────────────────────────────────────────────

# ── Stage 1: Build React frontend ──────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build
# Output: /build/dist/

# ── Stage 2: CPU-only Python runtime ──────────────────────────
FROM python:3.11-slim AS cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps — CPU stack (no torch-cuda, no vLLM)
COPY requirements-cpu.txt ./
RUN pip install --no-cache-dir -r requirements-cpu.txt

# Copy application code
COPY app/ app/
COPY configs/ configs/
COPY scripts/ scripts/
COPY pyproject.toml ./
COPY .env.example .env.example

# Copy the pre-built React frontend
COPY --from=frontend-build /build/dist/ frontend/dist/

# Create data/uploads and outputs dirs
RUN mkdir -p data/uploads data/processed outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Stage 3: CUDA runtime (default target) ─────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS default

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Install CUDA-aware Python deps
COPY requirements-cuda.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r requirements-cuda.txt

# Copy application code
COPY app/ app/
COPY configs/ configs/
COPY scripts/ scripts/
COPY pyproject.toml ./
COPY .env.example .env.example

# Copy the pre-built React frontend
COPY --from=frontend-build /build/dist/ frontend/dist/

# Create data/uploads and outputs dirs
RUN mkdir -p data/uploads data/processed outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
