# WatchLog — Production Dockerfile
# Multi-stage build: keeps image lean by excluding dev deps

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed to compile some Python packages (opencv-headless, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime libs for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

# Create data dirs
RUN mkdir -p /app/clips /app/models

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT','8000') + '/health')"

CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"
