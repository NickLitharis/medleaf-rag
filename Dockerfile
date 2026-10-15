# MedLeaf retrieval API + indexer image.
#
# Dependencies are installed here, not inherited from a host virtualenv: a clean clone
# gets a working torch/Qdrant/FastAPI stack without a GPU or a pre-existing venv.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# CPU-only torch first: the default PyPI wheel drags in ~2 GB of CUDA libraries that
# this service never uses.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[api,retrieval]" \
    && python -c "import fastapi, qdrant_client, sentence_transformers, medleaf; print('deps ok')"

ENV MEDLEAF_DATA_DIR=/app/data \
    HF_HOME=/app/data/models

EXPOSE 8000

CMD ["uvicorn", "medleaf.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
