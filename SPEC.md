# MedLeaf: specification

Status: v3, 2026-11-23. Replaces the RAG-service spec, which is kept at the git tag
`rag-service`.

## 1. Question

When E5 embeddings of EU medicine information are split into clusters and a query searches
only its nearest clusters, does e-means (arXiv:2611.09501) find the right sections as often
as k-means, and is it faster or does it fit the vectors more tightly?

## 2. Scope

In scope:
- Parse and chunk the EMA product-information PDFs (`src/medleaf/ingest`).
- Embed the chunks and the dev questions with the pinned multilingual E5 model, on Kaggle.
- Cluster the vectors with k-means and e-means, with several seeds.
- Score each clustering with Recall@5 from `src/medleaf/eval/metrics.py`, against an
  exhaustive search.

Out of scope: a served search or answer API, an LLM, a vector database, safety classifiers,
fine-tuning, new embedding models, and any use of the held-out test split.

## 3. Data

- `data/medicines.yaml`: 40 medicines, an EMA PDF URL and SHA-256 per language.
- `eval/datasets/retrieval-v1.jsonl`: 126 questions with gold sections, 63 dev and 63 test.
  `eval/splits.json` assigns whole medicines to train, dev or test.
- Embeddings: `intfloat/multilingual-e5-base` at a pinned revision, 768 dimensions.
- PDFs whose checksum has changed since the pin are left out, with the questions that name
  them. They are listed in the Kaggle run's `manifest.json`.

## 4. Method and decision rule

See `experiments/e-means/PLAN.md`. The settings (k, passes, batch size, cluster counts
searched) are fixed before results are seen. e-means is worth pursuing only if its Recall@5
is within one question of k-means at every cluster count and it is faster or has lower
error. A null result is acceptable.

## 5. Limits

- The e-means here is simplified: no step annealing and no dead-cluster revival.
- Scoring is dense-only, so its Recall@5 is not comparable with any hybrid-search figure.
- With about 43 usable dev questions, one question is about 2.3 percentage points.
- The corpus is EMA text; attribution is in `NOTICE.md`.
