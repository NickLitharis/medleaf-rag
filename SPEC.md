# MedLeaf RAG — Specification

Bilingual (Greek / English) question answering over official EU medicine information,
with cited answers, medical safety guardrails, and an evaluation gate that blocks
regressions in CI.

- **Owner:** Nikos Litharis
- **Status:** Draft v2 — 2026-10-15
- **Repo:** `github.com/NickLitharis/medleaf-rag` (public, MIT licence for code)

---

## 1. Problem

Patients, carers and pharmacy or medical students often need quick, reliable answers
about a medicine: *"What are the common side effects of Ozempic?"* or *"Αλληλεπιδρά το
Eliquis με την ιβουπροφαίνη;"* ("Does Eliquis interact with ibuprofen?"). Generic chatbots answer from memory, mix up medicines, and
cannot point to the official text. The official text exists, but it is a long PDF.
Most open-source RAG demos are English-only and give no proof that they work.

## 2. Goal

A self-hostable API that answers Greek or English questions about a fixed set of
medicines, using only their official EU product information. It cites the exact document
section for every claim. It refuses when the sources do not cover the question, and it
redirects personal medical questions to a professional. It **proves its quality with an
automated evaluation that runs on every pull request**.

Portfolio goal: publicly demonstrate the skills most requested in job ads (see §14),
and continue the medical domain of the BSc thesis (text generation from medical images)
in a text-only, production-style form.

## 3. Scope

### In scope
- Corpus: product information for ~40 medicines, each in Greek and English (§5).
- Ingestion: download, parse PDF, chunk by numbered section, embed, index.
- Hybrid retrieval: dense + sparse (BM25) search, fused, then reranked.
- Answer generation with inline citations, streamed over SSE (Server-Sent Events).
- Medical safety layer: personal-advice and emergency detection (§9).
- Evaluation set (~120 questions) and a scoring CLI.
- LoRA fine-tune of the embedding model, compared against the base model.
- CI on GitHub Actions: lint, tests, evaluation gate.
- Tracing with Langfuse; experiment tracking with MLflow.
- One-command local run with Docker Compose.

### Non-goals
- **No diagnosis, no personal dosing, no treatment advice.** The system explains what
  the official document says, nothing more. Every answer carries a disclaimer.
  It is a portfolio demo, not a medical device and not for clinical use; the README
  says so first.
- No patient data of any kind. Users may still type personal details, so question text
  is not sent to Langfuse unless `TRACE_CONTENT=true` (on only in local dev), and the
  README asks users not to enter personal health data.
- No web UI. FastAPI's built-in `/docs` page and `curl` examples are enough.
- No user accounts, auth, or multi-tenancy.
- No cloud deployment (Project 2 covers Kubernetes/Terraform).
- No corpus updates at runtime; the corpus is versioned and rebuilt offline.
- No code, data, prompts, or test cases from any employer or client project.

## 4. Success criteria

| # | Criterion | Target |
|---|-----------|--------|
| S1 | `docker compose up` gives a working API from a fresh clone, using the prebuilt index snapshot | < 15 min including model downloads, no manual steps besides `.env` |
| S2 | Retrieval Recall@5 on the `test` split, best configuration | ≥ 0.85 |
| S3 | Citation precision (cited sections that are correct) | ≥ 0.90 |
| S4 | Correct refusal on unanswerable questions | ≥ 0.90 |
| S5 | Correct safety redirect on the ~40-question safety set (§11) | **1.00** for `emergency`, ≥ 0.95 for `personal` |
| S6 | Greek and English scores within 5 points of each other | Recall@5 and citation precision |
| S7 | Fine-tuned embeddings vs base model | Measured and reported, win or lose |
| S8 | A PR that lowers Recall@5 by more than 1 question fails CI | Demonstrated in a real PR |
| S9 | README results table generated from the evaluation output | Not hand-written |

Targets S2–S4 are starting guesses. Lock them after the first baseline run (M3) and
record the change in `docs/decisions.md`. The S5 emergency target is not negotiable.

## 5. Data

**Source:** EMA (European Medicines Agency) EPAR product information. EMA publishes it
in all EU languages, Greek included, for every centrally authorised medicine. Each
product-information PDF contains:
- **SmPC** (Summary of Product Characteristics, Annex I), written for professionals, with
  fixed numbered sections, e.g. 4.1 indications, 4.2 dosing, 4.3 contraindications,
  4.4 warnings, 4.5 interactions, 4.6 pregnancy, 4.8 side effects, 6.4 storage.
- **Package leaflet** (Annex IIIB), written for patients, with sections 1–6.

The same numbering is used in every language, which gives stable, language-independent
chunk IDs.

**Medicine list:** ~40 widely used, centrally authorised medicines across therapy areas,
for example: diabetes (Ozempic, Jardiance, Forxiga, Trulicity), anticoagulants (Eliquis,
Xarelto), cardiology (Entresto), neurology (Lyrica), immunology (Humira), vaccines
(Comirnaty). The final list is a committed file, `data/medicines.yaml`.

Note: older common medicines such as paracetamol or ibuprofen are usually authorised
per country, not by EMA, so they are **not** in the corpus. Questions about them are
good "not in corpus" refusal tests.

- **Finding the documents:** no guessed URLs. EMA's
  [Download medicine data](https://www.ema.europa.eu/en/medicines/download-medicine-data)
  page offers tables of all centrally authorised medicines, updated every night. The
  download script reads the medicine's EMA page link from that table, then the English
  and Greek product-information PDF links from the medicine's page. M1 confirms which
  table and column hold the link; `medicines.yaml` stores the resolved URLs.
- **Licence (checked 2026-10-15):** EMA's legal notice allows reproduction and
  distribution, in whole or in part, in any format, for commercial and non-commercial
  use, provided EMA is acknowledged as the source **in each copy**. This does not cover
  content whose copyright belongs to a third party.
  - So the Qdrant snapshot (which contains chunk text) may be shared, but the snapshot
    metadata, every citation and the README must name EMA as the source.
  - In M1, check that the product-information PDFs carry no third-party copyright notice.
- **Storage:** PDFs are downloaded by a script into `data/raw/` (gitignored). The repo
  stores only `medicines.yaml` and a SHA-256 checksum per file, so builds are
  reproducible. Each record keeps the document's revision date. EMA updates these PDFs;
  a checksum mismatch means a new corpus version (the script reports it), not a failure.
- **Size estimate:** 40 medicines × 2 languages × ~60 sections → roughly 4–6k chunks.

## 6. Architecture

```
            ingestion (offline, CLI)                      serving (FastAPI)
 EMA PDFs ─► parse ─► chunk ─► embed ─► Qdrant ◄── hybrid search ◄── safety check ◄── POST /v1/ask
                               (dense + sparse)        │
                                                       ▼
                                              rerank (cross-encoder)
                                                       │
                                                       ▼
                                         LLM via LiteLLM ─► SSE stream
                                                       │
                                               Langfuse traces
```

Components, each a Docker Compose service:
- `api` — FastAPI app (async).
- `qdrant` — vector database with dense and sparse vectors.
- `ollama` — optional local LLM for free runs (`--profile local`).
- `langfuse` — optional tracing (`--profile tracing`). Self-hosted Langfuse v3 needs
  Postgres, ClickHouse, Redis and object storage; if that is too heavy for a laptop, point
  the app at the Langfuse Cloud free tier instead (same SDK, env vars only).
- `mlflow` — optional experiment tracking UI (`--profile ml`).

A prebuilt Qdrant snapshot is attached to each GitHub Release, so a fresh clone can run
the API without re-embedding the corpus. `medleaf ingest` rebuilds it from scratch.

## 7. Tech stack

| Concern | Choice | Rejected alternative (why) |
|---------|--------|----------------------------|
| API | FastAPI, async, Pydantic v2 | Flask (no native async/typing) |
| PDF parsing | PyMuPDF, with layout-aware table extraction | Plain text extraction (loses side-effect tables) |
| Vector DB | Qdrant (dense + sparse, RRF fusion built in) | pgvector (already on CV; Qdrant adds a named product) |
| Dense embeddings | `intfloat/multilingual-e5-base` | OpenAI embeddings (paid, can't fine-tune) |
| Sparse / BM25 | Qdrant sparse vectors, Greek-aware tokenizer | Elasticsearch (another heavy service) |
| Reranker | `BAAI/bge-reranker-v2-m3` (multilingual) | No reranker (lower precision) |
| LLM access | LiteLLM (OpenAI, Azure, Ollama behind one interface) | Direct SDKs (provider lock-in) |
| Fine-tuning | sentence-transformers + PEFT (LoRA), PyTorch | Full fine-tune (bigger, slower, less to show) |
| Evaluation | Own harness + RAGAS for faithfulness | RAGAS only (no citation/refusal/safety metrics) |
| Tracing | Langfuse | LangSmith (SaaS only) |
| Experiments | MLflow | Weights & Biases (SaaS account) |
| Tests | pytest, pytest-asyncio, httpx | — |
| Lint | ruff | — |
| Packaging | uv, `pyproject.toml` | pip + requirements.txt |
| CI | GitHub Actions | — |

## 8. Ingestion and chunking

1. **Parse** each PDF into: medicine → document (`smpc` | `leaflet`) → numbered section →
   subsection. Skip Annex II and the labelling annex (packaging text, little value).
2. **Chunk by numbered section, not by token count.** One chunk = one section, e.g. SmPC
   4.3. Sections over ~400 tokens are split at subsection or paragraph boundaries.
3. **Tables** (mostly side-effect frequencies in SmPC 4.8) become one line per row, e.g.
   `Nausea — very common (≥1/10)`, and stay in the same chunk as their heading.
4. **Metadata per chunk:** `medicine`, `active_substance`, `atc_code`, `lang`, `doc`
   (`smpc`|`leaflet`), `section`, `title`, `source_url`, `revision_date`,
   `corpus_version`.
5. **Chunk ID** is stable and language-independent, e.g. `eliquis:smpc:4.5`. The Greek
   and English versions of the same section share the ID, which lets evaluation score
   cross-lingual retrieval.
6. **Index** dense and sparse vectors into one Qdrant collection per corpus version.
   - e5 models require prefixes: `passage: ` when embedding chunks, `query: ` when
     embedding questions. Missing prefixes silently lower recall; cover this with a test.
   - Sparse vectors are computed in the app (accent-stripped, lowercased tokens) and
     stored with Qdrant's `idf` modifier, so Qdrant applies BM25-style weighting.

Acceptance: parser tests check, for every medicine and language, that the expected SmPC
sections (4.1–4.9, 6.4) and leaflet sections (1–6) were found, and that the Greek and
English section lists match.

## 9. Safety, retrieval and generation

**Safety check (before retrieval).** A small classifier (LLM call with a fixed prompt,
plus keyword rules as a backstop) labels each question:
- `info` — a general question about what the document says, including "Can X be taken
  with Y?" → continue; the answer reports what the document says and ends with a
  reminder to check with a pharmacist.
- `personal` — the question gives the user's own situation (age, dose, symptoms,
  condition) or asks for a decision for them, e.g. "I'm on 5 mg and missed a dose,
  should I double it?" → no answer from the documents; fixed message to ask a doctor
  or pharmacist.
- `emergency` — overdose, severe reaction, suicidal intent → fixed message with
  emergency numbers: 112 (EU-wide, answered in Greek, English and French); in Greece
  also 166 (EKAB ambulance) and the Poison Information Centre, 210 779 3777 (checked
  2026-10-15). Suicidal intent also gets the Greek suicide-prevention line 1018
  (Klimaka; verify in M4). Never calls retrieval.

When unsure, the classifier picks the safer label. Emergency keyword rules (Greek and
English) run first and override the LLM label. All three paths are logged.

**Retrieval and generation (for `info`).**
1. Detect the question language (simple heuristic: Greek script ratio).
2. Hybrid search: top 30 dense + top 30 sparse, fused with RRF (Reciprocal Rank Fusion).
   If the question names medicines, filter to those medicines. Names are matched against
   brand names and active substances in both languages from `medicines.yaml`
   (e.g. Eliquis / apixaban / απιξαμπάνη), accent-insensitive.
   Search both languages; the answer is written in the question's language. Hits for the
   same chunk ID in both languages are merged into one, and the citation URL points to the
   version in the question's language.
3. Rerank the top 20 fused results with the cross-encoder; keep the top 6. (Reranking
   more on CPU is too slow; the cut-off is measured in M4.)
4. If the top rerank score is below a threshold (tuned on `dev`), **refuse** without
   calling the LLM.
5. Prompt the LLM with the 6 chunks, each labelled with its citation key, and require a
   citation after every claim, e.g. `[Eliquis SmPC 4.5]`. The prompt also tells the model
   to refuse when the chunks do not support an answer. This second check is needed for
   wrong-premise questions, which often retrieve relevant chunks and pass step 4.
6. Post-check: drop citation keys the model invented (not in the 6 chunks) and log them
   as a defect in the trace.

Every stage is a plain function with typed inputs and outputs, so evaluation can run
retrieval alone without the LLM.

## 10. API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/ask` | Question → streamed answer (SSE) |
| `POST` | `/v1/search` | Question → ranked chunks, no LLM (debug + eval) |
| `GET` | `/v1/medicines` | Medicines in the corpus, with revision dates |
| `GET` | `/healthz` | Process is alive |
| `GET` | `/readyz` | Qdrant reachable and collection loaded |

`POST /v1/ask` request: `{ "question": str, "lang": "el"|"en"|null, "top_k": int = 6 }`

SSE events, in order:
- `safety` — `{ label: "info"|"personal"|"emergency" }`.
- `retrieval` — list of chunk IDs and scores (only for `info`).
- `token` — answer text, repeated.
- `citations` — list of `{ key, medicine, doc, section, url, revision_date }`.
- `done` — `{ trace_id, latency_ms, refused: bool, disclaimer: str }`.
- `error` — `{ code, message }` instead of the remaining events if something fails.

Failure handling at the boundary:
- LLM timeout (30 s) or provider error → `error` event; the retrieval result is still logged.
- Safety classifier fails → treat as `personal` (safe default), never as `info`.
- Qdrant down → `/readyz` fails and `/v1/ask` returns `503`.
- Client disconnects mid-stream → the LLM call is cancelled.

## 11. Evaluation

### Dataset
- **~120 questions**, split 50/50 Greek/English, written by hand. An LLM may draft
  questions, but every one is checked against the source text before it is added.
- Stored as versioned JSONL in `eval/datasets/`, one record per question:
  `id, lang, question, category, gold_chunk_ids[], expected_label, must_refuse, notes`.
- **Categories:**
  - single-section lookup (indications, storage, pregnancy…)
  - side effects with frequency ("how common is…")
  - interactions and contraindications
  - patient vs professional wording (leaflet vs SmPC for the same fact)
  - cross-medicine comparison (two medicines in one question)
  - cross-lingual (Greek question, gold also matched in English)
  - not in corpus (~10%, e.g. paracetamol)
  - wrong premise (~5%, e.g. a side effect the document does not list)
- **Safety set:** a separate ~40-question file (`eval/datasets/safety.jsonl`, 50/50
  Greek/English): emergency, personal and borderline `info` questions, scored on
  `expected_label`. It is kept apart so S5 rests on more than a handful of questions.
- **Medicine splits.** Every medicine is assigned to exactly one of three groups, in a
  committed file `eval/splits.json`:
  - `train` — used only to generate fine-tuning data (§12); no evaluation questions.
  - `dev` — questions used to tune thresholds and prompts, and run by the CI gate.
  - `test` — questions reported in the README; run only at milestones, never tuned on.
- A question belongs to `dev` or `test` only if **all** its medicines are in that group.
  Comparison questions use two medicines from the same group. Rough sizes:
  16 train / 12 dev / 12 test medicines.
- **Recall@k:** a question is a hit when at least one gold chunk *per named medicine* is
  in the top k. The same fact often sits in both the SmPC and the leaflet, so both IDs
  can be gold. A chunk ID matches in either language (IDs are shared, §8).
- With ~60 questions per split, one question is ~1.7 points. Report counts next to
  percentages so small changes are not over-read.

### Metrics
| Level | Metric |
|-------|--------|
| Safety | Label accuracy, with **missed `emergency`/`personal` counted separately** |
| Retrieval | Recall@5, Recall@10, MRR@10 (Mean Reciprocal Rank) |
| Citations | Precision and recall against gold chunk IDs |
| Answer | Faithfulness (RAGAS), refusal accuracy, invented-citation rate |
| Ops | p50/p95 latency, tokens and cost per question |

All metrics are reported overall and per language and category.

### Runner
`medleaf eval --split test --config configs/best.yaml` writes
`eval/results/<run>.json` and logs the run to MLflow. A second command renders the README
results table.

## 12. Fine-tuning

- **Training data:** synthetic (question, chunk) pairs generated by an LLM from
  **`train` medicines only** (§11), ~3k pairs, both languages, in both patient and
  professional wording. Check a 50-pair random sample by hand. A CI test fails if any
  training pair references a `dev` or `test` medicine.
- **Method:** LoRA adapter on `multilingual-e5-base` with sentence-transformers + PEFT,
  `MultipleNegativesRankingLoss`.
- **Hardware:** a free Colab or Kaggle GPU; the notebook lives in `training/`. The adapter
  is published to the Hugging Face Hub so CI and Docker do not need a GPU.
- **Comparison:** base vs fine-tuned, with and without the reranker → a 2×2 results table.
  Report the result even if fine-tuning does not help; that is a valid finding.
- Every run is logged to MLflow (params, metrics, adapter hash).

## 13. CI (GitHub Actions)

| Workflow | Trigger | Steps | Secrets |
|----------|---------|-------|---------|
| `ci.yml` | every PR / push | ruff, pytest (unit + API with fakes), safety keyword-rule tests | none |
| `eval-retrieval.yml` | every PR | build index from cached corpus, run retrieval eval (search + rerank, `configs/best.yaml`) on `dev`, compare with `eval/baseline.json`, **fail if Recall@5 drops by more than 1 question (~2 points)** | none |
| `eval-full.yml` | manual + weekly | safety + full answer eval with the LLM on `dev`; `test` only when run manually at a milestone; **fails on any missed emergency** | `OPENAI_API_KEY` |

- Retrieval eval needs no LLM, so the PR gate is free and deterministic.
- The gate uses `dev`, not `test`: gating every PR on `test` would slowly tune the
  system to the test set and make the README numbers optimistic.
- The built Qdrant snapshot is cached in Actions, keyed by corpus checksum + model hash.
- Updating `eval/baseline.json` is its own PR, so improvements are explicit.

## 14. Skills covered (from job-tracker gap analysis)

| Job-ad skill | Where in this project |
|--------------|-----------------------|
| Named vector DB | Qdrant, hybrid dense + sparse |
| FastAPI / async Python | API layer, SSE streaming |
| PyTorch / Hugging Face | Embeddings, reranker, LoRA fine-tune |
| LoRA / PEFT | §12 |
| LLM evaluation (RAGAS) | §11 |
| Guardrails / responsible AI | Safety layer (§9), safety metrics |
| Observability (Langfuse), MLflow | Traces, experiment tracking |
| CI/CD (GitHub Actions), pytest | §13 |
| Docker / Compose | Local stack |

Not covered here, on purpose: Kubernetes, Terraform, LangGraph, MCP, Redis → Project 2.

## 15. Repository layout

```
medleaf-rag/
├── src/medleaf/
│   ├── ingest/        # download, PDF parse, chunk, index
│   ├── safety/        # classifier, keyword rules, fixed messages
│   ├── retrieval/     # hybrid search, rerank, refusal threshold
│   ├── generation/    # prompt, LiteLLM call, citation post-check
│   ├── api/           # FastAPI app, SSE
│   ├── eval/          # metrics, runner, report
│   └── config.py      # pydantic-settings, env vars only
├── configs/           # retrieval/generation configs (YAML)
├── data/medicines.yaml
├── eval/
│   ├── datasets/      # versioned JSONL
│   ├── splits.json    # medicine → train | dev | test
│   ├── baseline.json  # dev-split gate values
│   └── results/       # gitignored except the latest summary
├── training/          # LoRA notebook + script
├── tests/
├── docs/decisions.md  # short decision log
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example       # variable names only
└── README.md          # includes the medical disclaimer and EMA attribution
```

## 16. Milestones

Each milestone is one or more small PRs with its acceptance check.

| M | Deliverable | Acceptance |
|---|-------------|------------|
| M1 | Repo skeleton, CI (`ci.yml`), download + parse for 5 medicines EN+EL | Section-coverage tests pass; side-effect tables readable; licence wording confirmed |
| M2 | All ~40 medicines, hybrid search, `/v1/search`, Docker Compose | `curl` returns relevant chunks for 5 sample questions in each language |
| M3 | Eval dataset v1 + runner + retrieval gate | Baseline recorded; a deliberately broken PR fails CI (S8) |
| M4 | Safety layer, reranker, refusal, `/v1/ask` with SSE + citations | S3, S4, S5 measured; invented citations stripped and logged |
| M5 | Langfuse tracing, RAGAS faithfulness, `eval-full.yml` | One trace per request with safety, retrieval and LLM spans |
| M6 | LoRA fine-tune + 2×2 comparison | Adapter on HF Hub; results table in README |
| M7 | Release with Qdrant snapshot, README polish, demo GIF, CV line | S1 verified from a fresh clone on a clean machine |

## 17. Risks

| Risk | Mitigation |
|------|------------|
| Users treat answers as medical advice | Safety layer, disclaimer on every answer, S5 targets, fixed redirect messages |
| Reranker too slow on CPU | Rerank only the top 20; report p95 latency; fall back to a smaller reranker if p95 > 5 s |
| PDF layout varies (tables, two-column pages, Greek encoding) | Section-coverage tests per medicine and language (M1); fail loudly on missing sections |
| Greek and English PDFs differ in revision date | Store `revision_date` per language; flag pairs that differ and exclude them from cross-lingual questions |
| Greek BM25 tokenization is weak (accents, inflection) | Normalize accents and lowercase; measure sparse-only recall in M2; drop sparse if it adds nothing |
| Synthetic training data leaks into evaluation | Split by medicine; CI test for overlap |
| LLM costs | Retrieval gate needs no LLM; full eval is manual or weekly with a small model |
| Small local LLM is weak in Greek | Use it only for free demos; report results with the API model |
| Corpus goes out of date | Show `revision_date` in every citation; rebuilds are a documented command |
| Scope creep | Non-goals in §3; anything new goes to a "later" list in the README |

## 18. CV line (fill in after M6)

> **MedLeaf RAG** — Bilingual Greek/English RAG over official EU medicine information
> (FastAPI, Qdrant, Hugging Face, LoRA, RAGAS, Langfuse, GitHub Actions). Reached [X]
> Recall@5 and [Y] citation precision on a 120-question evaluation set, with a safety
> layer that redirected 100% of emergency questions in a 40-question safety set and a CI gate
> that blocks retrieval regressions; LoRA-tuned embeddings changed Recall@5 by [Z] points.
