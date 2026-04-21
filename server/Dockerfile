# ML Debug Env — Dockerfile
# Build context root: ml_debug_env/  (same dir as openenv.yaml)
# Usage from ml_debug_env/:
#   docker build -f server/Dockerfile -t ml-debug-env:latest .
#   docker run -e GROQ_API_KEY=... -p 8000:8000 ml-debug-env:latest

FROM python:3.10-slim

# ── system packages ────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# ── working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps: torch CPU wheel first (keeps layer small and cacheable) ───────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cpu

# ── Other Python deps (openenv-core, fastapi, openai, etc.) ───────────────────
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
        git+https://github.com/meta-pytorch/OpenEnv.git && \
    pip install --no-cache-dir \
        fastapi>=0.115.0 \
        "uvicorn[standard]>=0.24.0" \
        python-multipart \
        aiofiles \
        "openai>=1.0.0" \
        "pydantic>=2.0"

# ── Application code ───────────────────────────────────────────────────────────
# Copy in order: models first (imported by both server and client)
COPY models.py      /app/models.py
COPY client.py      /app/client.py
COPY __init__.py    /app/__init__.py
COPY server/        /app/server/

# ── Environment ────────────────────────────────────────────────────────────────
# PYTHONPATH=/app so that `from models import ...` works from server/
ENV PYTHONPATH=/app
# Tell grader.py which Python exe to use for subprocess code execution
ENV PYTHON_EXEC=/usr/local/bin/python

EXPOSE 8000

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start server ───────────────────────────────────────────────────────────────
# Run from /app/server so relative imports within server/ still work
WORKDIR /app/server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]