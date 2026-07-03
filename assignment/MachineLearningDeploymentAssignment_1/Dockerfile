# ─────────────────────────────────────────────
# FashionMNIST Classifier — Production Docker Image
# ─────────────────────────────────────────────
# Build:  docker build -t fashion-mnist-api .
# Run:    docker run -p 8080:8080 -e API_KEY=your-key fashion-mnist-api
# ─────────────────────────────────────────────

FROM python:3.11-slim

LABEL org.opencontainers.image.title="FashionMNIST Classifier API"
LABEL org.opencontainers.image.description="REST API for FashionMNIST image classification"
LABEL org.opencontainers.image.version="1.0.0"

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Cloud Run provides PORT env var; default to 8080
ENV PORT=8080
ENV MODEL_DIR=/app/models

WORKDIR /app

# ── System dependencies ──────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────
# Copy requirements first (better layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────
COPY main.py .
COPY models/ ./models/

# ── Security: non-root user ──────────────────
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# ── Health check ─────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# ── Run ──────────────────────────────────────
EXPOSE ${PORT}

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --log-level info"]
