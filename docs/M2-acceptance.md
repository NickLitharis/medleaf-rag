# M2 acceptance — hybrid retrieval over 40 medicines (+ the open rebuild gate)

**Status: acceptance NOT met.** The corpus, the hybrid retrieval stack, the API and the
ten-question ``curl`` run are real and reproducible, but the *index currently serving
those queries was built before this milestone's chunking and metadata fixes*, and a
measured gate says it is not good enough to claim M2 complete:

* **727 of 2772 indexed chunks exceed the model window** (up to 164,861 tokens against a
  512-token window), so dense vectors for those chunks only see the first 512 tokens and
  the sparse index contains text the model never reads. Measurement and worst offenders:
  `docs/m2-window-report.json` (produced by `medleaf index window-report --json` against
  the real Qdrant payloads, after payload annotation).
* **5 of the 10 sample questions put the correct SmPC section in the top 5**, not 5 per
  language as M2 requires. The five failures are caused by the oversized chunks: a giant
  `*:smpc:10` block matches almost every Greek query through the sparse channel.

The code fixes are implemented and tested; the **rebuild is the outstanding gate** (see
"Required to close M2"). No completion is claimed for M2 in `docs/milestones.json`.

---

## 1. Corpus — 40 medicines, EN + EL (met)

* `data/corpus-request.txt` records every requested EMA name.
* 40 medicines across the ATC areas of the SPEC: endocrinology (Ozempic, Jardiance,
  Forxiga, Trulicity, Rybelsus, Mounjaro, Victoza), antithrombotics (Eliquis, Xarelto,
  Pradaxa, Lixiana, Brilique), cardiology/lipids (Entresto, Repatha, Praluent),
  neurology/MS (Lyrica, Aimovig, Tecfidera, Ocrevus, Tysabri), immunology (Humira,
  Enbrel, Remicade, Stelara, Cosentyx, Dupixent, Rinvoq), oncology (Keytruda, Opdivo,
  Imfinzi, Herceptin, Tagrisso, Ibrance), vaccines (Comirnaty, Shingrix), respiratory
  (Fasenra, Kaftrio), urology/oncology (Nubeqa), haematology/ophthalmology (Soliris,
  Eylea).
* **80/80 documents downloaded, language-verified and SHA-256 pinned.** The five M1
  entries in `data/medicines.yaml` are byte-identical to the previous revision, so no
  existing pin was touched.
* **Documented rejections** (never counted as parsed or indexed):
  * `Perjeta` — the `/el/` product-information URL served **Estonian** bytes
    (sha256 `e663bc517f96…`, text begins `RAVIMI OMADUSTE KOKKUVÕTE`). The downloader's
    language re-check caught it, the entry was dropped and `data/raw/perjeta/` deleted.
  * Not in EMA's bulk medicines report: `Crestor`, `Symbicort`, `Trelegy`, `Xeljanz`,
    `Ferinject`, `Toujeo`, `Xolair`, `Vabysmo`, `Aranesp`.
* Official requests are cached under `data/cache/ema/` (24 h TTL for the bulk report) and
  rate-limited; `medleaf download` retries HTTP 429/5xx with exponential backoff. The
  first run used four workers and hit 429s; the documented default is now two.

Reproduce:

```bash
PYTHONPATH=src python -m medleaf.cli resolve --continue-on-error \
    $(grep -v '^#' data/corpus-request.txt | tr '\n' ' ')
PYTHONPATH=src python -m medleaf.cli download --workers 2
```

## 2. Model and dependencies (met)

| Item | Value |
|---|---|
| Model | `intfloat/multilingual-e5-base`, revision `d128750597153bb5987e10b1c3493a34e5a4502a` (pinned in `MEDLEAF_EMBED_REVISION`) |
| Weights | `model.safetensors` (`model_kwargs={"use_safetensors": True}`), `trust_remote_code` never enabled |
| Dimension | 768 dense; sparse vectors with Qdrant `Modifier.IDF` |
| Cache | `data/models` (1.1 GB, gitignored, reused by the next milestone) |
| Prefixes | `passage: ` for chunks, `query: ` for questions, enforced by `medleaf.retrieval.embed.prefixed` and covered by tests that fail if a mismatched prefix is passed |
| Local deps | Python 3.13 venv with `--system-site-packages`; uv resolved its own torch/transformers wheels inside the venv. The Docker image installs CPU-only torch and `.[api,retrieval]` independently (verified: `docker build` exit 0) |

## 3. Index build (built, but from pre-fix code)

```bash
docker run -d --name medleaf-qdrant-test -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 qdrant/qdrant:v1.15.5
PYTHONPATH=src python -m medleaf.cli index build
```

* 2772 points across 80 medicine-language pairs, 0 missing sources, runtime **1566 s**
  (~26 min: ~5 min parse, ~21 min e5 embedding on CPU).
* `data/index/manifest.json`: `corpus_version=medicines.yaml:9ddd9754264c`,
  `dense_dim=768`, `embed_revision=d12875059715…`, `total_chunks=2772`.
* Collection shape verified live: `dense` (768, COSINE) plus `sparse` with
  `Modifier.IDF`.

## 4. Measured correctness gate (NOT met)

```bash
PYTHONPATH=src python -m medleaf.cli index window-report --json
# collection=medleaf_chunks points=2772 over_window=727 limit_tokens=496
# worst: 164861 tokens comirnaty:smpc:10 | 156079 enbrel:smpc:10 | 145220 xarelto:smpc:10 ...
# all 40 medicines have at least 11 over-window chunks
```

Diagnosis:

1. **Oversized-paragraph bug (fixed in code).** The previous splitter only cut *between*
   paragraphs and emitted a single oversized paragraph whole, so a reflowed section with
   no blank lines became one chunk of up to 164 k tokens. `split_long_text` now packs
   table rows, paragraphs, sentences and whole words into parts that fit the real
   tokenizer window (`ChunkBudget` fed by `Encoder.count_tokens`), and
   `tests/test_m2_retrieval_unit.py` proves the text is partitioned, never dropped.
2. **Parser span leak (root cause, still open).** Those giant blocks are SmPC `10`,
   leaflet `6` and `5.2` sections containing far more text than the section itself
   (Comirnaty's English cache holds 903 KB with no `ANNEX II` marker at all, i.e. the
   annex boundary was never found and the document tail was attributed to the last
   known heading). M1 validated the boundary detector on five medicines; the wider
   corpus shows it does not generalise. Fixing this is parser work in
   `medleaf/ingest/pdf.py` (`find_annex_boundaries` / `_smpc_annex_end`), outside M2's
   scope, and it must land before the rebuild — splitting annex text into thousands of
   chunks would make the index worse, not better.
3. **Metadata (met, payload-only).** `medleaf index annotate` attached the SPEC section
   8 `active_substance`/`atc_code` values to all 2772 existing points *without*
   re-embedding, and the manifest now records them plus `missing_metadata` (empty: EMA
   supplied both fields for all 40 medicines). `split_sections` is still empty because
   that field is written by a build, not an annotation.

### Required to close M2

```bash
# 1. Fix the SmPC/leaflet end detection in src/medleaf/ingest/pdf.py and re-run the M1
#    corpus tests (they pin the five known-good medicines).
MEDLEAF_M1_CORPUS=1 PYTHONPATH=src python -m pytest tests/test_m1_acceptance.py -q
# 2. Rebuild the affected medicines with the tokenizer-aware chunker. The targeted form
#    prunes those medicines' points first, so no stale chunk IDs survive, and re-embeds
#    only what was asked for. A full rebuild (drop --only) is ~26 min on CPU.
PYTHONPATH=src python -m medleaf.cli index build --no-recreate --only <affected slugs>
# 3. Prove the gate is closed (must print over_window=0) and re-run the acceptance tests.
PYTHONPATH=src python -m medleaf.cli index window-report --json
MEDLEAF_M2_CORPUS=1 MEDLEAF_QDRANT_URL=http://127.0.0.1:6333 \
    PYTHONPATH=src python -m pytest tests/test_m2_corpus_index.py -q
# 4. Re-run the ten sample questions and update this document.
bash .tmp/curl-queries.sh   # same commands, see section 5
```

## 5. Ten sample questions with `curl` (5 EN, 5 EL)

Run against the live API (`uvicorn medleaf.api.app:create_app --factory`), all with
`medicine=<slug>&doc=smpc&limit=5`. Section order is the top-5 the API returned; the
expected section is underlined by the **hit** column.

| # | lang | question | expected | top-5 sections | hit |
|---|------|----------|----------|----------------|-----|
| en-1 | en | What are the contraindications of Eliquis? | 4.3 | 10, 4.7, 5.1, 4.4, 4.5 | no |
| en-2 | en | How should Humira be stored? | 6.4 | 10, **6.4**, 5.1, 4.4, 6.5 | yes |
| en-3 | en | Which side effects of Ozempic are very common? | 4.8 | 4.1, 5.1, **4.8#1**, 6.6, 2 | yes |
| en-4 | en | Can Xarelto be used during pregnancy? | 4.6 | **4.6**, 10, 4.2, 4.4, 4.7 | yes |
| en-5 | en | What is Jardiance used for? | 4.1 | 4.7, 5.1, **4.1**, 4.4, 4.6 | yes |
| el-1 | el | Ποιες είναι οι αντενδείξεις του Eliquis; | 4.3 | 10, 1, 4.2, 5, 5.1 | no |
| el-2 | el | Πώς πρέπει να φυλάσσεται το Humira; | 6.4 | 10, 6, 5.1, 4.4, 4 | no |
| el-3 | el | Ποιες ανεπιθύμητες ενέργειες του Ozempic είναι πολύ συχνές; | 4.8 | 1, 6.6, 4.2, 6.3, 6.5 | no |
| el-4 | el | Επιτρέπεται η χρήση Xarelto στην εγκυμοσύνη; | 4.6 | 10, 1, 4.2, 4.4, 4 | no |
| el-5 | el | Σε ποιες ενδείξεις χρησιμοποιείται το Jardiance; | 4.1 | **4.1**, 5.1, 8, 4.6, 4.4 | yes |

**Result: 5/10 (5 EN, 0 EL of the 5 EL).** The English column passes because the giant
chunks are less competitive there; the Greek column fails for the reason in section 4.
Full JSON responses are reproducible with `.tmp/curl-queries.sh` (kept in the worktree,
not committed).

Provenance carried by every hit (verified on `en-1`):

```json
"source_url": "https://www.ema.europa.eu/en/documents/product-information/eliquis-epar-product-information_en.pdf",
"revision_date": "2026-03-25",
"corpus_version": "medicines.yaml:9ddd9754264c",
"active_substance": "apixaban",
"atc_code": "B01AF02"
```

Negative controls (all verified live):

| Control | Result |
|---|---|
| `/readyz` before any manifest existed | HTTP 503, `"index manifest … is missing; run \`medleaf index build\`"` |
| `/v1/search` before any index existed | HTTP 503 `{"error": "index_missing"}` (never an empty 200) |
| `/v1/search` with an unknown `corpus_version` | HTTP 503 `index_missing` for the filtered count — an unindexed corpus version is an error, not an empty success |
| `/v1/search` with `lang=en` | only `"lang":"en"` results; unfiltered search reaches both languages |
| `/v1/search` with `active_substance=apixaban` | only `"medicine":"eliquis"` results |
| `test_language_filter_is_exact_and_the_negative_control_proves_it` | asserts the filter excludes the other language *and* that removing it changes the result set |

## 6. Tests, lint and types

| Check | Result |
|---|---|
| `python -m ruff check .` | clean (whole repo) |
| `pytest -q` (offline fast suite, in-memory Qdrant) | 54 passed, 13 skipped, 28 deselected |
| `MEDLEAF_QDRANT_URL=… pytest tests/test_m2_qdrant.py` | 11 passed (real Qdrant server) |
| `MEDLEAF_M1_CORPUS=1 MEDLEAF_SMOKE_PDF=… pytest -q` | see `docs/milestones.json` M2 evidence for the final counts |
| `mypy` on the new modules | clean; the pre-existing `fitz`/`yaml` stub and `ema.py:240` errors that M1 already recorded are unchanged |
| Negative control (manual) | the language-filter test fails when `build_filter` ignores `lang`, then passes again once restored — the check can fail |
| `docker build -f Dockerfile -t medleaf-rag:local .` | exit 0 (independent CPU torch + `.[api,retrieval]` stack) |

## 7. Resource and runtime limits

* Index build 26 min on 16 CPU cores; peak memory ~2 GB (torch + e5-base, batch 32).
* Model download 1.1 GB into `data/models`; corpus PDFs ~250 MB in `data/raw`.
* Qdrant test container bound to `127.0.0.1:6333/6334`; the API to `127.0.0.1:8000`.
  Unrelated containers (`skillpulse-agent-postgres-1`, `hindsight`, `taxis-db`) were
  never touched, and global Docker settings were not changed.
* The e5 tokenizer emits a `token indices sequence length is longer than the specified
  maximum` warning while the window report measures the oversized chunks — that warning
  is exactly the truncation this milestone measured.

## 8. Not claimed

* No claim that the current index is complete or truncation-free; 727 over-window chunks
  are measured and reported.
* No claim that split-section parts are language-aligned: part boundaries differ between
  Greek and English for long sections (recorded per section in `split_sections` after a
  build).
* No clinical validation, no evaluation metrics (M3), no reranker, no answer generation.
* No claim that EMA's annex text is correctly excluded — section 4.2 records the opposite.
