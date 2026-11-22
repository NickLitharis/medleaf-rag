# Rust rewrite plan

Status: planning only; implementation is not authorized by this document.

## Goal and boundary

Replace the repository-owned Python application with Rust while preserving its current Greek/English medicine-information workflows: resolve, download, parse, index, search, evaluate, and serve cited answers with safety redirects. Keep Qdrant, the pinned E5 model, EMA corpus, OpenAI-compatible provider protocol, and existing browser interface. Preserve provenance and EMA attribution.

A language rewrite is not evidence of faster embedding inference or better retrieval. Do not combine it with the e-means experiment, new embeddings, reranking, sparse-ranking changes, fine-tuning, tracing, or other unfinished SPEC milestones. Measure any performance claim separately against the Python reference on the same hardware and inputs.

Groundwork guidance applied: planning-principles (evidence and gates), engineering-principles (one cohesive application), evolution-principles (behavior and stored-data compatibility), testing-principles (independent contract checks), security-principles (safe failures and dependency boundaries).

## Evidence and present limits

- `pyproject.toml`: core dependencies include HTTPX, Pydantic, PyMuPDF and PyYAML; retrieval adds Sentence Transformers, NumPy and Qdrant; serving adds FastAPI/Uvicorn.
- `README.md`: implemented browser demo, GET/POST search, buffered and validated answer output, safety-provider failure behavior, and direct OpenAI-compatible calls differ from older parts of `SPEC.md`.
- `docs/decisions.md`: direct provider HTTP is intentional; evaluation uses Recall@5 hit counts and rejects scored-count changes. Some safety descriptions are historical and must be reconciled with current code.
- `docker-compose.yml`: Qdrant v1.15.5; loopback ports; shared host data; separate one-shot indexer; API does not rebuild the index on startup.
- `.github/workflows/ci.yml` and `eval-retrieval.yml`: offline checks, opt-in real corpus acceptance, real Qdrant retrieval gate, cached snapshots and model assets.
- `docs/M2-acceptance.md`: records past first-copy/contents-list parser failures and token-window fixes. These are regression cases, not fresh validation of this checkout.
- Local inspection: Rust is installed under `/home/marocos/.cargo/bin` with a stable toolchain, but that directory is not on PATH. `data/` contains the medicine manifest and request list, but no raw PDFs or model/index cache. System Python lacks the reference dependencies. Isolated tooling and downloaded checksum-pinned assets are needed before claiming inference or full-corpus compatibility.

## Proposed architecture

One Cargo package with a shared library and `medleaf` binary. Modules follow existing responsibilities rather than creating microservices: configuration/models; ingestion; retrieval/index management; generation/safety; evaluation; HTTP and CLI adapters. Keep the current HTML/JavaScript page instead of rewriting it for aesthetic consistency.

Candidate dependencies, not yet compatibility-validated: `clap` for CLI, `tokio`/`axum` for serving, `reqwest` for provider and EMA HTTP, `serde` for persisted and wire types, and a maintained YAML parser for the existing manifest. Use a Qdrant client or its HTTP API only after proving compatibility with v1.15.5 and existing hybrid-query/alias behavior. Pin the selected Rust toolchain and dependencies.

Two implementation choices must be proved before committing to the rest of the rewrite:

1. **E5 inference:** evaluate a native Rust backend, with Candle as a candidate, using the exact pinned model/tokenizer, prefixes, pooling and normalization. ONNX Runtime is an alternative only if the model conversion, artifact pinning, numerical behavior and runtime dependency are acceptable. No Python subprocess or inference service may silently become the final implementation.
2. **PDF extraction:** evaluate a maintained Rust PDF interface backed by a suitable native engine, with PDFium as a candidate. Require Greek Unicode text, reading-order geometry and side-effect table behavior. A plain text extractor is not an equivalent replacement for PyMuPDF. Review native-engine distribution, licensing and version pinning. If adequate extraction cannot be demonstrated, stop the rewrite at this gate and present the evidence rather than dropping ingestion.

Native libraries are compatible with a Rust application but are not a promise of a single static binary. Do not select crates or deployment artifacts solely because they avoid Python.

## Execution sequence and acceptance gates

### 0. Freeze the reference and contract inventory

Prepare an isolated Python environment using the existing documented install commands. Record tool/dependency versions, effective non-secret configuration, model revision, corpus hash, Qdrant version, manifests and snapshot identifiers. Restore or download checksum-matching corpus assets without silently changing the pinned corpus. Capture current API schemas, CLI help/options/exit codes and deterministic serialized artifacts.

Run existing checks and record actual counts/skips: `python -m pytest -q`, `python -m ruff check .`, opt-in corpus tests when assets exist, and `medleaf eval --split dev --baseline eval/baseline.json` against the real pinned Qdrant. Existing failures are reference findings, not reasons to weaken Rust assertions.

**Gate:** a checked-in contract inventory and attributable reference fixtures cover the externally consumed interfaces and index-shaping behavior. No credentials or user health information enters fixtures. Resolve contradictions between specification, tests and current behavior explicitly.

### 1. Prove model and PDF feasibility

Build bounded throwaway Rust probes, not a parallel production stack. Compare tokenizer IDs/counts on Greek/English, Unicode edge cases and window boundaries. Compare vector shape, finiteness, norm and cosine/L2 differences on a fixed representative sample. Set and document numerical tolerances before interpreting results; validate ranking effects rather than requiring bitwise float equality.

Compare extracted sections and table rows against the pinned Python reference, including bilingual medicine variants, repeated full-document copies, leaflet contents lists and numeric medical statements. Document every mismatch; inspect the actual EMA page where the reference may itself be wrong. Sweep all 80 pinned PDFs before the parser gate closes.

**Gate:** demonstrated native inference and complete corpus extraction with no unapproved loss of sections, table values, attribution or passage-window safety. Remove probes after retaining the reproducible checks and evidence.

### 2. Port data, configuration and ingestion

Implement the shared domain types, configuration parsing, manifest handling, resolver/downloader, section recognition, table transformation, deterministic chunking and parse cache. Preserve effective environment precedence, validation, SHA-256 handling, URLs/revisions, stable IDs and serializer behavior where hashes or caches depend on it. Preserve bounded EMA concurrency and failure handling.

**Gate:** Rust produces compatible manifest/cache/chunk identities and metadata for reference fixtures and all corpus documents. Every known parser regression is covered. If an artifact must intentionally change, version it and specify rebuild/recovery before continuing.

### 3. Port retrieval and index publication

Implement tokenizer/name matching and sparse vectors before index builds. Port native dense inference, Qdrant collection definitions, filtering, RRF queries, result deduplication, readiness validation, index status/window diagnostics and snapshot-compatible metadata. Build into an isolated candidate collection; do not publish it until all validation finishes.

**Gate:** Python-built data can be read by Rust where compatibility is promised; Rust-built data can serve equivalent searches. Exercise missing/stale/incompatible manifests and unavailable Qdrant as failures, never successful empty responses. Interruption of a candidate build leaves the live alias serving the old valid index. No passage exceeds the effective model input window.

Run the unchanged dev dataset and baseline gate. Keep the existing scored-count requirement and allowance of at most one lost Recall@5 hit; also report per-question changes, Recall@10, MRR and language breakdown. Use the held-out test split only at the migration milestone, not for repeated tuning.

### 4. Port safety, generation, HTTP, CLI and evaluation

Reuse the shared Rust retrieval implementation across API and CLI. Preserve the existing routes, query/body validation, response fields, status/error codes, SSE events and actual buffering semantics. Keep emergency rules ahead of provider/retrieval work, personal escalation and safety-provider failure handling. Preserve extractive/prose modes, structured-output requirements, source-ID validation, citation/number guards, refusal text, disclaimers and provider timeout semantics.

Keep the existing page and prove it works through the real Rust HTTP server. Port all existing CLI subcommands and evaluation dataset/gate semantics; do not declare parity because only search and ask work.

**Gate:** black-box contract fixtures pass against both implementations for deterministic paths. Real Qdrant exercises protocol/serialization; a controllable provider exercises malformed responses, timeouts and SSE fragmentation without relying on live nondeterministic model text. Optional live-provider checks remain separate and accurately scoped. Verify emergency questions perform no provider/retrieval call, and no unvalidated model answer is exposed to the browser.

### 5. Integrate delivery, cut over, and remove Python

Create a multi-stage Rust image including required model/PDF runtimes and certificates. Preserve Compose service names, persistent storage and loopback exposure; replace Python entrypoints and update documentation and CI. Cache keys must cover Cargo manifests/lockfile, Rust index-shaping code, model/tokenizer artifacts and the corpus pin. Preserve real-Qdrant evaluation and opt-in corpus coverage.

Run `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, real-service retrieval and full-corpus acceptance. These are proposed Rust commands, not commands that have run in this checkout. Smoke the actual CLI and browser from the release image with and without LLM credentials.

**Gate:** all requested workflows run without application Python, completed contract/corpus/retrieval gates, documented operational instructions, and tested recovery. Then remove obsolete Python source/tests/packaging, Python image steps, invocation examples and CI wiring in the same clean cutover. Port useful tests first; do not retain implementation-specific tests merely to preserve their count.

## Recovery and completion

Before cutover, back up the active collection snapshot and matching local manifest/cache/configuration, and retain a runnable reference release image. Run comparison builds against separate collections and data directories. Publish only validated artifacts; do not point the old process at a new incompatible manifest.

A binary rollback is safe only while its data/artifacts remain compatible. For intentional format changes, restore the matching snapshot and local state together or document forward repair. Prove recovery on disposable storage before touching the live alias. Do not delete the reference snapshot/image until the Rust smoke and evaluation gates have passed and recovery is demonstrated.

Done means the full application and delivery path are migrated, not merely a compiling server. No fake inference, omitted PDF tables, reduced safety handling, permanent Python bridge or unrelated ranking change counts as completion.

## Contract details confirmed by scoped source review

### Serving, CLI and evaluation

- Preserve `GET /`, `/v1/medicines`, `/healthz`, `/readyz`, GET/POST `/v1/search`, and POST `/v1/ask`. Search/ask reject unknown fields; ask accepts 1–1000 question characters and `top_k` 1–10. Preserve HTTP 422 validation and `detail.error` on HTTP 503 failures, not just route names (`src/medleaf/api/app.py`).
- Emergency/personal SSE sequence is `safety`, `token`, `citations`, `done`. Informational answers emit `safety`, `retrieval`, followed by an answer sequence or `error`. Classifier/retrieval failures occur before SSE begins; generation/grounding failures occur inside SSE. The current checked answer is buffered and emits at most one answer `token` event (`api/app.py`, `generation/answer.py`).
- Classifier retries once; failed classification redirects only when a personal keyword rule matched, otherwise HTTP 503 `safety_unavailable`. Emergency rules skip the classifier and retrieval (`src/medleaf/safety/classifier.py`). Preserve the bilingual fixed emergency messages; independently revalidating medical contact policy would be a separate explicit change.
- Port every command: `resolve`, `download`, `check`, `tables`, `index build/status/annotate/window-report`, `search`, and `eval`. Include stdout/stderr, exit statuses, default file writes and `download --no-pin`. Resolver/downloader can rewrite the corpus manifest, so reference preparation must not silently repin it (`src/medleaf/cli.py`).
- The committed dev baseline is 48/57 Recall@5 hits: 63 questions, six unscored. Part IDs match parent sections; multi-medicine hits require a gold section for each named medicine. Test that a one-hit loss passes, a two-hit loss fails, and a changed scored count fails (`src/medleaf/eval/metrics.py`, `runner.py`, `eval/baseline.json`).
- `done.refused` currently means no valid citation was used, not independently proven refusal accuracy. The evaluator is retrieval-only; there is no implemented general safety/answer-quality evaluation gate. Do not present those SPEC targets as already achieved.
- Reuse behavior from `tests/test_m2_api.py`, `test_m4_answer.py`, `test_m4_safety.py`, `test_m3_eval.py`, `test_cli_check.py` and `test_m4_faithfulness.py`. Existing CLI coverage is sparse; add black-box coverage of actual command side effects during migration. Client-disconnect cancellation is a SPEC promise without a dedicated test found in this review: characterize the reference and explicitly decide the required behavior rather than assuming parity.
- Provider calls send questions outside the process. Keep existing credential aliases and local-only deployment defaults. Do not add authentication/cloud exposure as hidden rewrite scope. Current answer errors can include upstream response text; define a minimal redacted error contract if secret leakage is demonstrated, recording that as an intentional safety correction rather than copying sensitive bytes into golden fixtures.

### Ingestion, embeddings and persisted index

- Preserve YAML `Medicine`/`DocumentRef` shapes and both `en`/`el` documents. Default corpus version hashes the exact manifest bytes; explicit non-`dev` override wins. Reformatting YAML can therefore change corpus identity even without semantic edits (`ingest/store.py`, `retrieval/index.py`, `retrieval/pipeline.py`).
- `data/parsed/<slug>_<lang>.jsonl` is misleadingly named: it contains one JSON object, not newline-delimited records. Preserve that format or explicitly invalidate it. Cache identity includes `medleaf-parser-3`, PDF SHA-256 and budget/model/revision; corpus version is restamped rather than an invalidation input.
- Preserve root-based path defaults, not just environment variable names. The index manifest has strict schema 1 with model, sparse tokenizer, collection, corpus and completeness metadata. Unknown fields and incompatible tokenizer identity must not silently pass.
- Logical chunk IDs are language independent: `<slug>:<doc>:<section>`, with `#1..N` for parts. Physical IDs are UUIDv5 over `<lang>:<chunk_id>` with namespace `1a2f6e2c-9c0f-4d5b-9f1c-7f1e4c2b8a10` (`retrieval/index.py`). Pin exact cross-language and cross-process examples before porting.
- Sparse tokenizer identity is `medleaf-tokenizer-2`: Unicode lowercase, NFD/Mn removal, final-sigma folding, alphanumeric tokens, minimum length two, one longest-first Greek suffix removal with minimum three-character stem, and query-only stopword removal. IDs use big-endian 32-bit BLAKE2b of UTF-8 tokens, with sorted sparse indices and count weights (`retrieval/text.py`). Match complete vectors, not only a few search results.
- E5 is pinned to `intfloat/multilingual-e5-base@d128750597153bb5987e10b1c3493a34e5a4502a`, with 768-dimensional normalized vectors. Default max sequence length is 512; chunking reserves 16 tokens and measures the exact serialized header plus body. Do not mix changed-runtime query vectors with old passage vectors until compatibility is demonstrated. A failing numerical/ranking gate requires a separately versioned full re-embedding, followed by retrieval acceptance.
- Parser parity includes first-complete-copy selection, title-validated headings, mixed Greek/Latin annex glyphs, soft-hyphen reflow, language detection, and section-owned SmPC 4.8 tables. Existing table handling uses geometry and bold spans to distinguish headings from reactions; plain text matching is insufficient (`ingest/pdf.py`, `ingest/chunk.py`, `ingest/pipeline.py`, `sections.py`).
- Qdrant uses named cosine dense vectors, sparse IDF, exact filters and server-side RRF. Rust SDK support must be exercised against v1.15.5. Full builds use candidate collections and alias swapping; targeted builds delete selected medicine points before upserting, so they are not inherently rollback-safe.
- **Required cutover correction:** current full-build publication swaps the alias and deletes the old collection before writing the local manifest (`retrieval/pipeline.py`). Do not copy this crash window into Rust or claim the reference already provides atomic cross-store publication. Stage durable candidate metadata, retain the previous collection and matching metadata, serialize publication, and validate manifest/alias agreement before serving. Crash-injection checks must exercise every publication boundary; a mismatch fails closed and recovery restores a matching pair. Delete old state only after successful verification. This is an explicit operational correction, not a ranking change.

### Concrete corpus and retrieval evidence to run during implementation

```bash
MEDLEAF_M1_CORPUS=1 python -m pytest tests/test_m1_acceptance.py tests/test_m2_boundaries.py -q
python -m pytest tests/test_m2_retrieval_unit.py -q
MEDLEAF_QDRANT_URL=http://127.0.0.1:6333 python -m pytest tests/test_m2_qdrant.py -q
MEDLEAF_M2_CORPUS=1 MEDLEAF_QDRANT_URL=http://127.0.0.1:6333 PYTHONPATH=src python -m pytest tests/test_m2_corpus_index.py -q
medleaf eval --split dev --baseline eval/baseline.json
```

These are existing Python reference commands, not completed checks. The real-model suite exercises all 40 medicines, ten top-five smoke queries, language filters, input windows and cross-language similarity. Reject skipped/empty integration runs as evidence. Port these behaviors to Rust, adding missing independent checks for configuration precedence, slug/YAML/cache compatibility, synthetic Greek marker cases, publication crashes and interrupted targeted builds.

## Feasibility evidence collected

### Native E5 embedding smoke: passed

Reproduction: `bash experiments/rust-embeddings/run.sh`. Detailed evidence and model/source hashes: `experiments/rust-embeddings/results.json`.

The probe loads the exact pinned safetensors directly into Candle 0.11.1 XLM-RoBERTa on CPU/F32 and compares against the existing Python `E5Encoder` using torch 2.8.0+cpu. All 15 raw inputs and 19 padded batch rows match token IDs/counts and attention masks, including mixed Greek/English and 496/512/513-token boundaries. Maximum vector component difference is `8.79e-7`, minimum cosine is `0.99999999997`, and maximum L2-norm difference is `1.12e-7`; all meet the predeclared tolerances. Four queries have identical complete rankings over eight sample passages.

This closes native E5 smoke feasibility only. It does not establish full-corpus retrieval parity, Qdrant compatibility, production throughput, or a reason to change the model. Keep the stage-3 real-index and held-out acceptance gates.

### PDFium text/layout smoke: passed; native table parser remains unproven

Evidence and reproduction inputs: `experiments/rust-pdf/results.json` and `run_comparison.py`. The Rust probe uses `pdfium-render` 0.9.6 with PDFium 158.0.8125.0, compared against PyMuPDF 1.28.2. All downloaded PDFs match their existing manifest checksums; EMA attribution is retained in the results.

The run covers Xarelto, Eliquis and Humira in English and Greek: six documents, 1,672 pages. Page counts and dimensions match. Aggregate normalized character recall is 0.99984–0.99998, Greek-character recall is 1.0, and every text-bearing page exceeds the predeclared 90% character-recall floor. These character-multiset metrics do not prove reading order or semantic equivalence. The Python-accepted first-copy headings are found on the same pages in all six PDFs.

The diagnostic covers 30 SmPC 4.8 tables and 996 nonempty cells. Rust-extracted glyphs assigned to **PyMuPDF-provided cell rectangles** achieve 1.0 alphanumeric recall and bold-cell agreement; exact cell-text match is 0.876–0.986. Using the existing **Python row flattener**, frequency-association recall is 1.0, but exact flattened-row recall is 0.645–0.917. The independent eight-check Xarelto English frequency oracle passes for both reference and diagnostic outputs.

This is not an independent Rust parser: native table detection, first-copy/section selection, row flattening and frequency association have not been ported. The remaining 74 PDFs have not been exercised. Before the full rewrite's parser gate closes, implement those operations in Rust without reference cell geometry, resolve the observed text/row differences, and run the complete pinned corpus. The engine can supply useful Unicode/layout/font primitives; table-parser parity remains an open migration risk.

Reproduce the PDF probe:

```bash
uv venv /tmp/medleaf-pdf-venv --python 3.13
uv pip install --python /tmp/medleaf-pdf-venv/bin/python -r experiments/rust-pdf/requirements.txt
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH=src CARGO_HOME=/tmp/medleaf-pdf-cargo
export CARGO_TARGET_DIR=/tmp/medleaf-pdf-target CARGO_BUILD_JOBS=1
/tmp/medleaf-pdf-venv/bin/python experiments/rust-pdf/run_comparison.py --slugs xarelto,eliquis,humira
```

### Decision after these probes

Native E5 inference is a viable candidate for the Rust rewrite based on the bounded numerical smoke. PDFium is a viable extraction primitive, not a drop-in replacement for the current table/parser pipeline. Do not start a complete cutover on the assumption that PDF parsing is solved. No production application code was replaced, and neither probe establishes a performance advantage.

### Independent native Rust tables: tested; overall gate failed

The next probe now detects ruled tables, reconstructs cells/bold flags, and flattens rows in Rust (`experiments/rust-pdf/src/tables.rs`). The candidate binary receives only the PDF path and pinned PDFium library, not Python cell rectangles or reference rows. Python still selects the first-copy SmPC 4.8 pages for evaluation; native first-copy/section selection is not demonstrated.

Reproduction, using the retained isolated environment and pinned assets:

```bash
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH=src CARGO_HOME=/tmp/medleaf-pdf-cargo
export CARGO_TARGET_DIR=/tmp/medleaf-pdf-target CARGO_BUILD_JOBS=1
/tmp/medleaf-pdf-venv/bin/python experiments/rust-pdf/run_native_tables.py
```

Final observed output: six documents / 1,672 pages processed, 44 reference-owned first-copy 4.8 pages, 29 native tables versus 30 reference tables, and five of six documents passing. Xarelto Greek, Eliquis English/Greek and Humira English/Greek pass. Xarelto English fails: four versus five tables, cell-content precision 0.9040 and recall 1.0000. One matched page-11 reference table has three columns while its native match has seven; structural separation/content contamination remains unresolved. Overall status is `FAIL`, and the command exits 1.

All six documents achieve 1.0 frequency-association precision/recall under the recorded comparison. The independent eight-check Xarelto English oracle and reviewed Eliquis English/Greek cases pass. Negative controls reject swapped frequencies, missing/empty tables, and an indication-column swap that preserves the row frequency multiset.

Corrections made during this experiment: labelled rows now retain all header/value context; superscript geometry no longer moves footnotes before reaction words in the tested cases; non-frequency tables remain visible rather than being silently discarded. Evaluation also now normalizes multiline Greek frequency labels before matching: otherwise `Όχι` followed by a line break and `συχνές` was incorrectly recognized as common rather than uncommon. Regression controls cover multiline English/Greek labels. A 95% aggregate column gate allowed a two-cell swap among 144 associations, so an additional stricter zero-assignment-mismatch gate was added; no acceptance threshold was loosened.

The first native run detected 23/30 tables and passed zero documents. After extractor corrections, four documents passed but the new swap control exposed the evaluation weakness; the final evaluator correction produces five passes and valid negative controls. Earlier reports are retained in `/tmp/medleaf-native-table-initial-results.json` and `/tmp/medleaf-native-table-second-results.json`; the final reproducible report is `experiments/rust-pdf/native-table-results.json`.

**Decision:** native table processing is feasible on these samples but is not yet a complete replacement. Do not cut over ingestion until the Xarelto English structural failure is resolved, all 80 pinned PDFs are exercised, and native section/first-copy selection is independently validated. This bounded experiment is complete with a failing compatibility result, not a claim that parser parity passed. Production application code is unchanged.

### Xarelto English segmentation correction: six-document gate passed

The remaining failure had two independent geometry causes in `experiments/rust-pdf/src/tables.rs`:

- Co-linear ruling segments from separate tables shared one connectivity-graph vertex, bridging the whitespace between tables. Each disconnected span is now a separate vertex.
- Candidate filtering required three horizontal boundaries, excluding a valid single-row, four-column header table on Xarelto English page 11. Two horizontal boundaries are now sufficient; existing size and grid-reconstruction checks remain.

Both regressions were reproduced before the fixes: separate grids collapsed to one bounding box, and a single-row table produced no candidate. The two Rust unit tests now pass, including vertically stacked and side-by-side grids. No medicine or page special cases were added, and evaluation thresholds are unchanged.

Verification:

```bash
cargo test --release --locked --manifest-path experiments/rust-pdf/Cargo.toml
/tmp/medleaf-pdf-venv/bin/python experiments/rust-pdf/run_native_tables.py
```

Use the environment exports from the preceding reproduction instructions. Observed results: two unit tests passed; all six documents passed; all 30 reference tables and 352 reference rows were detected. Xarelto English now has 5/5 tables with cell-content and frequency-association precision/recall all 1.0. All six documents have those same 1.0 scores; reviewed medical oracles, strict column-assignment gates and negative controls pass. The final report is `experiments/rust-pdf/native-table-results.json`, and the command exits 0.

This supersedes the previous sample-level `FAIL` decision, not the full-corpus gate. Only six of 80 PDFs have been evaluated; Python still supplies evaluation-page selection. Native first-copy/section selection and the other 74 PDFs remain to be validated before ingestion cutover. Production Python remains unchanged.

### Expanded manifest run: 66 PDFs tested; full gate failed

The evaluator now accepts `--all` to select every medicine's English and Greek PDFs from the checksum-pinned manifest. Its default remains the six-document sample; acceptance thresholds and the native extractor were not changed for this measurement.

```bash
/tmp/medleaf-pdf-venv/bin/python -u experiments/rust-pdf/run_native_tables.py \
  --all --output experiments/rust-pdf/full-corpus-table-results.json
```

Use the existing environment exports above. The final command exited 1: 80 PDFs requested, 66 analyzed, 46 passed, 20 failed current gates, and 14 blocked by checksum mismatches. This includes the original six passing PDFs; of the remaining 74, 40 passed, 20 failed gates and 14 could not be tested. The run processed 9,156 pages and scored 349 Python-selected first-copy SmPC 4.8 pages, comparing 158 native tables against 164 reference tables (1,890 versus 1,965 rows). Reviewed medical oracles and negative controls pass.

Asset recovery preserved the manifest: the initial run analyzed only 29 PDFs because 37 requests were throttled and 14 live PDFs had changed. A paced retry recovered 25 matching PDFs; 12 previously throttled assets also proved to have changed checksums. Archive recovery found another 12 exact revisions (11 English, one Greek), bringing the checksum-verified cache to 66 PDFs. Archive URLs, attempted captures and hashes are recorded in `experiments/rust-pdf/archive-recovery-en.json` and `archive-recovery-el.json`. Rejected downloads were not substituted or repinned.

The 14 unrecovered pinned PDFs are `pradaxa/el`, `entresto/el`, `ocrevus/el`, `stelara/el`, `cosentyx/el`, `keytruda/el`, `opdivo/en`, `opdivo/el`, `imfinzi/en`, `imfinzi/el`, `comirnaty/el`, `shingrix/el`, `eylea/el`, and `soliris/el`. Live revisions have different hashes, and checked archive captures did not supply the exact originals. Completing those tests requires the original bytes or an explicitly approved new corpus version; the latter would not establish compatibility with the old pins.

The 20 gate failures are `rybelsus/en`, `rybelsus/el`, `lyrica/en`, `lyrica/el`, `ocrevus/en`, `tysabri/en`, `tysabri/el`, `remicade/en`, `remicade/el`, `stelara/en`, `dupixent/en`, `rinvoq/en`, `keytruda/en`, `herceptin/el`, `tagrisso/el`, `comirnaty/en`, `ibrance/el`, `fasenra/el`, `nubeqa/en`, and `nubeqa/el`. Exact per-document criteria, table matches and missing/unexpected associations are in the full report.

Do not interpret every failed gate as a demonstrated Rust medical error:

- `lyrica/en`, `tysabri/en`, `herceptin/el`, `fasenra/el`, and `nubeqa/en` fail only because the current row predicate recognizes zero frequency associations in both reference and candidate. Their reported empty-set frequency ratios are 1.0, but the gate correctly does not claim positive evidence from zero comparable rows.
- `lyrica/el` and `stelara/en` have no detected reference tables on the evaluated pages, while Rust detects two. This is an unresolved reference/detection coverage difference, not evidence that Rust necessarily invented tables.
- Other cases expose detection/content differences, differing flattened frequency rows, or column-assignment mismatches. Examples include Comirnaty English (1/9 tables), Rybelsus Greek (2/3), Ocrevus English (1/3), Nubeqa Greek (1/3), extra Remicade/Rinvoq tables, and a Tagrisso Greek column-association difference.

**Decision:** the six-document smoke passed, but corpus-wide parity has not. Keep ingestion cutover blocked. Resolve extractor differences and adjudicate unsupported reference layouts without weakening clinical invariants; obtain the 14 original revisions or explicitly version a replacement corpus. Native first-copy/page selection is still outside this experiment. Production code remains unchanged.

### Twenty available-corpus failures resolved

The final cached-corpus run passes all 66 checksum-available PDFs, including all 20 documents that failed the expanded run. This fixes both native extraction defects and demonstrated errors or gaps in the original Python comparison oracle; it does not claim byte-identical output from the uncorrected Python parser.

Native changes in `experiments/rust-pdf/src/main.rs` and `src/tables.rs`:

- Apply PDF path transformation matrices before reconstructing ruling geometry.
- Handle short junction gaps, shifted row rules, unruled text/frequency columns, single-row tables and page-edge rows bounded by matching outer rails. A one-sided protruding rule is not sufficient to invent a row.
- Preserve physical cell line boundaries, wrapped qualifiers, raised prefix/suffix markers and frequency-first/inline row layouts.
- Recover discretionary hyphens only when PDFium reports `FPDFText_IsHyphen` for its `U+0002` marker. Raw Unicode values remain in glyph metadata; numeric minus signs and unflagged control characters are not blindly rewritten.

Evaluator corrections in `run_native_tables.py`, `native-table-cases.json` and `tests/test_native_table_evaluator.py`:

- Score ordered, whitespace-insensitive source characters while retaining punctuation, numeric separators, qualifiers and reaction order; do not return to character bags.
- Parse split Greek frequency labels, ratio-qualified and multirow headers, preferred-term columns and verified aligned table continuations. Empty frequency evidence still fails.
- Keep complete, hash/page/provenance-bound source-table fixtures where PyMuPDF missed or malformed genuine tables, and record source-reviewed narrative/footer false detections separately. Preserve raw PyMuPDF metrics alongside adjudicated scoring.
- Preserve the Rybelsus Greek physical fragments on both pages and check the complete ordered logical frequency-column cell, including the following decreased-appetite entry. No physical cell is split merely because a footnote is followed by a capital letter.
- Retain positive medical checks and negative controls for dropped qualifiers, word order, decimal changes, missing/empty tables, swapped labels, wrong indication columns and lost cross-page fragments. Acceptance thresholds and the zero-column-assignment-mismatch requirement were not reduced.

Final verification, using the environment exports above:

```bash
cargo test --release --locked --manifest-path experiments/rust-pdf/Cargo.toml
PYTHONPATH=src /tmp/medleaf-pdf-venv/bin/python -m pytest tests/test_native_table_evaluator.py -q
/tmp/medleaf-pdf-venv/bin/python -u experiments/rust-pdf/run_native_tables.py \
  --cached-only --output experiments/rust-pdf/corpus-fixes-table-results.json
```

Observed: **21 Rust tests passed**, **17 evaluator tests passed**, and **66/66 available PDFs passed** the adjudicated table/content/frequency gates. The run processed 9,156 pages and evaluated 349 reference-selected SmPC 4.8 pages. All 158 adjudicated reference tables were detected; 164 raw PyMuPDF detections remain separately reported. Reference/native row counts are 2,025/2,024, so this is thresholded source-backed agreement, not a claim of identical serialization or every text character matching. Reviewed medical oracles, strict column-assignment checks and negative controls pass.

The report intentionally remains `FAIL` and the corpus command exits 1 because **14 of 80 requested pinned PDFs are still missing** in cached-only mode. These are the same 14 listed above, not extraction regressions. The completed fix covers all 20 available-corpus failures without hiding those coverage errors.

**Decision:** native table feasibility is established for the 66 available pinned PDFs under these checks. Full-80 validation and independent native section/first-copy selection remain unverified; do not treat this as a production ingestion cutover or clinical validation. Production Python and corpus pins remain unchanged.
