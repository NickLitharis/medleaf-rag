# M2 acceptance — hybrid retrieval over 40 medicines

**Status: acceptance met (repaired).** The 40-medicine corpus, the multilingual dense +
Qdrant sparse hybrid, `POST/GET /v1/search` and the ten-question `curl` run are real and
reproducible, and the previously failing gates are closed on the stored index:

* `medleaf index window-report --json` on the served collection: **8797 points,
  `over_window=0`** against the 496-token passage window (`docs/m2-window-report.json`).
* **10/10 sample questions put the expected SmPC section in the top 5** (5 EN, 5 EL),
  identical through the API `curl` run and `RetrievalService` tests.
* The M1 corpus coverage still passes, and the repaired parser returns every required
  SmPC/leaflet section for all 40 medicines in both languages.

The earlier "acceptance NOT met" state was real, but its recorded *cause* was wrong. This
document records the measured cause, the fix, and the evidence that closes it.

---

## 1. What was actually wrong (measured)

The pre-fix index held **2772 points, 727 of them over the model window** (worst
`comirnaty:smpc:10` at 164,861 tokens). The recorded cause was "the annex boundary is not
found". It is not: `find_annex_boundaries` finds Annex I/II/III for every one of the 80
PDFs, and Comirnaty's English parse *does* stop before `ANNEX II`. The real causes were:

1. **Repeated complete document copies.** An EMA "product information" PDF can contain
   several complete SmPCs (one per presentation/variant) inside Annex I and several
   package leaflets inside Annex III, all sharing the same section numbers. The old
   splitter kept only the *first* occurrence of each number, so everything after the
   first copy's `10. DATE OF REVISION` was attributed to `smpc:10` (and after the
   first leaflet's `6.` to `leaflet:6`). Comirnaty, for example, holds five such copies.
2. **The leaflet's numbered contents list.** A leaflet opens with "What is in this
   leaflet" listing `1.`..`6.` a line apart. Those were accepted as real headings, so the
   real leaflet body was attributed to the last contents entry (`6`).
3. **A short sparse-query signal.** Passages were embedded and sparse-indexed from the
   section body alone (no medicine/section title), so a question naming a section
   ("contraindications") could not match section `4.3` lexically, while a question's
   common words could be won by raw term frequency in long chunks.

Fix (single root-fix pass plus one bounded correction):

* `src/medleaf/ingest/pdf.py` now scans headings in document order, drops a numbered
  contents run, and stops at the first heading whose section number goes backwards
  **and** is confirmed as a restart (section `1` followed by `2` and `3`). Instruction
  steps numbered `1.` inside the body no longer end the copy. The same first-copy
  heading set now drives side-effect table page ownership.
* The baseline tolerance for joining a number and its title was widened (the Greek
  `4.7` number/title pair sits 1.6pt apart, so `4.7` was stranded); two Greek/English
  title variants were added (`Therapeutic indication`, `κατά την φύλαξη`).
* `build_sections` now serialises every passage with its medicine, document kind,
  section number and title, and measures *that exact serialised text* against the
  tokenizer budget, so the added context cannot push a chunk over the window. Dense and
  sparse inputs use the same text.
* Sparse queries drop function words and the medicine name that is already an exact
  filter (raw term frequency would otherwise let a long chunk win on `the`/brand-name
  repetition). The dense query keeps the full wording. This is query-side only, so it
  needs no re-embed.

## 2. Corpus — 40 medicines, EN + EL (met)

* `data/corpus-request.txt` records every requested EMA name.
* 40 medicines across ≥8 ATC areas (endocrinology, antithrombotics, cardiology/lipids,
  neurology/MS, immunology, oncology, vaccines, respiratory, urology/oncology,
  haematology/ophthalmology).
* **80/80 documents downloaded, language-verified and SHA-256 pinned.** The five M1
  entries are byte-identical to the previous revision.
* Documented rejections (never counted as parsed or indexed): `Perjeta` (the `/el/` URL
  served Estonian bytes, caught by the language re-check) and nine names absent from
  EMA's bulk report (`Crestor`, `Symbicort`, `Trelegy`, `Xeljanz`, `Ferinject`,
  `Toujeo`, `Xolair`, `Vabysmo`, `Aranesp`).
* Every required SmPC/leaflet section is present for all 40 medicines in both languages
  after the boundary fix (`tests/test_m2_boundaries.py`, opt-in with
  `MEDLEAF_M1_CORPUS=1`).

Reproduce:

```bash
PYTHONPATH=src python -m medleaf.cli resolve --continue-on-error \
    $(grep -v '^#' data/corpus-request.txt | tr '\n' ' ')
PYTHONPATH=src python -m medleaf.cli download --workers 2
```

## 3. Model and dependencies (met)

| Item | Value |
|---|---|
| Model | `intfloat/multilingual-e5-base`, revision `d128750597153bb5987e10b1c3493a34e5a4502a` |
| Dimension | 768 dense; sparse vectors with Qdrant `Modifier.IDF` |
| Parser / parse-cache version | `medleaf-parser-3` (`PARSER_CACHE_VERSION`) |
| Sparse tokenizer version | `medleaf-tokenizer-1` (`TOKENIZER_VERSION`) |
| Weights | `model.safetensors`, `trust_remote_code` never enabled; cached under `data/models` |
| Prefixes | `passage: ` for chunks, `query: ` for questions, enforced in `medleaf.retrieval.embed` |

## 4. Index build (repaired, measured)

```bash
docker run -d --name medleaf-qdrant-repair -p 127.0.0.1:6333:6333 qdrant/qdrant:v1.15.5
PYTHONPATH=src python -m medleaf.cli index build
```

* **8797 points** across 80 medicine-language pairs, 0 missing sources, `split_sections`
  = 891, `missing_metadata` = 0.
* Runtime ≈ 63 min on 16 CPU cores (parse + e5 embedding of the now-split corpus;
  the pre-fix 2772-point build was 26 min — long clinical sections such as `5.1`/`4.8`
  are now partitioned into window-sized parts instead of being silently truncated).
* `data/index/manifest.json`: `corpus_version=medicines.yaml:9ddd9754264c`,
  `embed_model=intfloat/multilingual-e5-base`,
  `embed_revision=d128750597153bb5987e10b1c3493a34e5a4502a`, `dense_dim=768`,
  `sparse_modifier=idf`, `tokenizer=medleaf-tokenizer-1`, `split_sections` populated.

## 5. Window gate (met)

```bash
PYTHONPATH=src python -m medleaf.cli index window-report --json
# collection=medleaf_chunks points=8797 over_window=0 clean=true limit_tokens=496
# worst: kaftrio:smpc:5.1#28 = 496 tokens (fits exactly)
```

Recorded in `docs/m2-window-report.json`. Every indexed chunk is readable by the model
without truncation; the longest part is exactly at the 496-token budget.

## 6. Ten sample questions (met, 10/10)

Run against the live API (`uvicorn medleaf.api.app:create_app --factory`), all with
`medicine=<slug>&doc=smpc&limit=5`. Section order is the top-5 the API returned. Evidence
script: `.tmp/repair-curl-evidence.sh` (kept in the worktree, not committed).

| # | lang | question | expected | top-5 sections | hit |
|---|------|----------|----------|----------------|-----|
| en-1 | en | What are the contraindications of Eliquis? | 4.3 | **4.3**, 4.2, 4.4, 4.2, 5.1 | yes |
| en-2 | en | How should Humira be stored? | 6.4 | **6.4**, 4.4, 6.5, 4.4, 4.2 | yes |
| en-3 | en | Which side effects of Ozempic are very common? | 4.8 | **4.8**, **4.8**, 4.4, 4.3, 4.1 | yes |
| en-4 | en | Can Xarelto be used during pregnancy? | 4.6 | **4.6**, 4.2, 4.2, 4.4, 4.5 | yes |
| en-5 | en | What is Jardiance used for? | 4.1 | 4.4, **4.1**, 4.7, 4.6, 4.2 | yes |
| el-1 | el | Ποιες είναι οι αντενδείξεις του Eliquis; | 4.3 | **4.3**, 4.2, 4.7, 4.2, 1 | yes |
| el-2 | el | Πώς πρέπει να φυλάσσεται το Humira; | 6.4 | **6.4**, 4.4, 4.4, 4.4, 6.5 | yes |
| el-3 | el | Ποιες ανεπιθύμητες ενέργειες του Ozempic είναι πολύ συχνές; | 4.8 | **4.8**, **4.8**, 6.6, 4.1, **4.8** | yes |
| el-4 | el | Επιτρέπεται η χρήση Xarelto στην εγκυμοσύνη; | 4.6 | **4.6**, 5.1, 4.2, 4.4, 4.4 | yes |
| el-5 | el | Σε ποιες ενδείξεις χρησιμοποιείται το Jardiance; | 4.1 | **4.1**, 4.7, 4.2, 4.6, 4.8 | yes |

**Result: 10/10.** The same ten queries pass through `RetrievalService` in
`tests/test_m2_corpus_index.py` (17 passed against the real index), so the `curl` surface
and the Python service return the same ranking.

### The earlier 5/10 curl evidence: diagnosed

The recorded `curl` run returned 5/10 while the Python rerun returned 8/10. That was not
retrieval randomness: on Windows the Greek question was passed as a **command-line
argument** to `curl`, and every non-ASCII byte became `?`. The API echoed the corrupted
input — `"query":"???e? e??a? ?? a?te?de??e?? t?? Eliquis;"` — so the API was asked a
different question. The repaired evidence script pipes each query through stdin
(`--data-urlencode "query@-"`), and the API now echoes the real UTF-8 question
(`"query":"Ποιες είναι οι αντενδείξεις του Eliquis;"`). The 8/10 Python rerun was the
truthful signal; the 5/10 curl evidence was a transport artifact.

Provenance carried by every hit (verified on `en-1`):

```json
"source_url": "https://www.ema.europa.eu/en/documents/product-information/eliquis-epar-product-information_en.pdf",
"revision_date": "2026-03-25",
"corpus_version": "medicines.yaml:9ddd9754264c",
"active_substance": "apixaban",
"atc_code": "B01AF02"
```

Negative controls (verified live): unknown `corpus_version` → 503 `index_missing`;
`lang=el` and `lang=en` each return only that language; `active_substance=apixaban`
returns only `eliquis`; the language-filter test fails if `build_filter` ignores `lang`
(deliberate break, then restore).

## 7. Tests, lint and types

| Check | Result |
|---|---|
| `ruff check .` (whole repo) | clean |
| Fast offline suite | 48 passed, 1 warning (before the rebuild) |
| `MEDLEAF_M2_CORPUS=1 MEDLEAF_QDRANT_URL=… pytest tests/test_m2_corpus_index.py` | **17 passed** |
| Full gated suite (`MEDLEAF_M1_CORPUS=1`, `MEDLEAF_M2_CORPUS=1`, real Qdrant) | **101 passed, 1 skipped** in 219.59s |
| `tests/test_m2_boundaries.py` (new) | 4 synthetic boundary/contents/fallback cases + opt-in 40-medicine coverage |
| Negative control | removing the stopword filter fails `test_query_sparse_vector_drops_function_words`, then passes once restored |
| mypy | pre-existing `fitz`/`yaml` stub and `ema.py:240` errors are unchanged; they are recorded, not hidden |

## 8. Not claimed / limitations

* No claim that split-section parts are language-aligned: part boundaries differ between
  Greek and English for long sections (`split_sections` records this per section).
* No clinical validation, no evaluation metrics (M3), no reranker, no answer generation.
* The index build (63 min, torch) is not run in CI; CI runs ruff and the fast suite, and
  the EMA corpus stays an explicit network opt-in. The rebuild and its evidence are
  reproduced locally per section 4.
* Standard EMA legal terms are recorded; this is not blanket legal or medical clearance.
* The pre-fix snapshot `data/index/snapshots/medleaf_chunks-8373388874623049-2026-10-15-15-07-48.snapshot`
  is preserved as a diagnostic artefact only and is never published.
