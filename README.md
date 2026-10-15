# MedLeaf RAG

Bilingual (Greek / English) question answering over official EU medicine
information, with cited answers and medical safety guardrails. The project is at
milestone **M2**: the corpus covers 40 centrally authorised medicines in both
languages and hybrid retrieval (dense + sparse) is served by `/v1/search`.
Answer generation, the safety layer and evaluation are later milestones; see
`SPEC.md`.

## Status

- **M1 — download + parse for five medicines (EN + EL):** complete. Evidence in
  `docs/M1-acceptance.md`.
- **M2 — 40 medicines, hybrid retrieval, `/v1/search`, Docker Compose:** code and corpus
  are done and tested, but the **acceptance is not yet met**: the index serving the
  queries was built before this milestone's chunking fix and `medleaf index
  window-report` measures 727 of 2772 chunks over the model window, so only 5 of the 10
  sample questions hit the right section. The measured gate, the root cause and the
  exact rebuild steps are in `docs/M2-acceptance.md`; status in `docs/milestones.json`
  is `in_progress`.
- **M3–M7:** not started.

## Setup

Python 3.11 or later. The retrieval stack (torch, sentence-transformers, Qdrant
client, FastAPI) is an extra, so a plain checkout stays light:

```bash
python -m pip install -e ".[dev]"          # lint, type checks and the fast tests
python -m pip install -e ".[api,retrieval]"  # index build + API
```

On a machine that already has a CPU torch, a virtualenv with
`--system-site-packages` reuses it instead of downloading torch again:

```bash
uv venv --system-site-packages --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[api,retrieval,dev]"
```

The embedding model is pinned by commit in `MEDLEAF_EMBED_REVISION`, downloaded to
`data/models` and recorded in the index manifest.

## Checks

```bash
PYTHONPATH=src python -m pytest -q          # fast tests, in-memory Qdrant
python -m ruff check .
```

The ten real EMA PDFs are not committed (`data/raw/` is ignored). To run the
corpus checks, download them first and then opt in:

```bash
PYTHONPATH=src python -m medleaf.cli resolve Eliquis Humira Jardiance Ozempic Xarelto
PYTHONPATH=src python -m medleaf.cli download
MEDLEAF_M1_CORPUS=1 PYTHONPATH=src python -m pytest tests/test_m1_acceptance.py -q
```

## Build the index and search

```bash
# 1. Resolve and download the 40-medicine corpus (network, respects EMA rate limits).
#    data/corpus-request.txt also lists the names EMA would not resolve, with reasons.
PYTHONPATH=src python -m medleaf.cli resolve --continue-on-error \
    $(grep -v '^#' data/corpus-request.txt | tr '\n' ' ')
PYTHONPATH=src python -m medleaf.cli download --workers 2

# 2. Run an isolated Qdrant and build the hybrid index
docker run -d --name medleaf-qdrant -p 127.0.0.1:6333:6333 qdrant/qdrant:v1.15.5
PYTHONPATH=src python -m medleaf.cli index build
PYTHONPATH=src python -m medleaf.cli index status

# 3. Search from the CLI or the HTTP API
PYTHONPATH=src python -m medleaf.cli search "What are the contraindications of Eliquis?" --lang en
PYTHONPATH=src uvicorn medleaf.api.app:create_app --factory --port 8000
curl "http://127.0.0.1:8000/v1/search?query=contraindications+of+Eliquis&lang=en&medicine=eliquis"
```

Every response carries full provenance (`chunk_id`, `source_url`, `revision_date`,
`corpus_version`). A missing index, an unreachable Qdrant or a model that cannot load
return HTTP 503 with an `error` code — the API never answers 200 with an empty list to
hide a broken index.

## Docker Compose

```bash
docker compose run --rm indexer        # build the index inside the stack
docker compose up -d api
curl "http://127.0.0.1:8000/v1/search?query=contraindications+of+Eliquis&lang=en"
```

Compose binds Qdrant and the API to loopback only, and the indexer runs as a one-shot
profile so `docker compose up` never rebuilds the index by accident. `data/` is mounted
from the host, so the corpus and the built index are not baked into images. Both
`indexer` and `api` use `pull_policy: build`, so every `run`/`up` rebuilds the image
from the current checkout first; a full index build fills a new collection and swaps
the `medleaf_chunks` alias only when it finishes, so the API keeps serving meanwhile.

## Medical disclaimer

This project is not medical advice and is not a substitute for a doctor or
pharmacist. Do not use it to decide on a dose, to stop or change a medicine, or in
an emergency. In an emergency in the EU, call 112.

## Licence and attribution

The code licence and the corpus licence are different. The medicine-information
PDFs and the text extracted from them are **copyright of the European Medicines
Agency (EMA)** and are not covered by any code licence; every copy must
acknowledge EMA as the source. The recorded EMA legal terms are in
`docs/M1-acceptance.md`. This repository claims no legal or clinical clearance.
