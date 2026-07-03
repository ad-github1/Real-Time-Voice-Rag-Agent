FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV VOICE_RAG_ENV=production
ENV VOICE_RAG_REASONER=template
ENV VOICE_RAG_RETRIEVER=bm25
ENV VOICE_RAG_DATA_DIR=/app/data/sample_docs
ENV VOICE_RAG_TRACE_DIR=/app/traces
ENV VOICE_RAG_INDEX_DIR=/app/data/index_cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/sample_docs ./data/sample_docs
COPY tests ./tests

RUN python -m pip install --no-cache-dir setuptools wheel \
    && python -m pip install --no-cache-dir -e ".[api,ingestion]" \
    && python -m compileall -q src

RUN mkdir -p /app/traces /app/outputs /app/data/index_cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "voice_rag_agent.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
