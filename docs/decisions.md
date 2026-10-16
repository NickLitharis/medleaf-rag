# Decisions

Short log of choices that are not obvious from the code. Newest last.

## 2026-10-16 — M3 evaluation

- **Splits are stratified by therapeutic area** (`eval/splits.json`): each area is spread
  over train/dev/test, so dev and test measure the same kind of question as train and not
  a domain shift. Same-substance medicines share a split (ozempic/rybelsus, semaglutide)
  so fine-tuning (§12) cannot leak through the substance.
- **Gold IDs name a section** (`eliquis:smpc:4.3`). Long sections are indexed as parts
  (`eliquis:smpc:4.8#3`); any part of a gold section counts as a hit.
- **Dataset v1** (`eval/datasets/retrieval-v1.jsonl`) has 126 questions, 63 per split,
  32/31 EN/EL. 57 per split are scored for retrieval. Not-in-corpus questions have no
  gold and are only used later for refusal. Every gold section was checked against the
  parsed EMA text. Every scored question names at least one dev or test medicine.
- **First baseline (dev, unfiltered query, top 10)**: Recall@5 48/57 (84.2%), Recall@10
  49/57, MRR@10 ≈0.62. Test: Recall@5 49/57 (86.0%), Recall@10 52/57, MRR@10 0.70. The
  weak spots are cross-medicine questions, Greek INN names and Greek-script brand names.
  S2 (≥ 0.85 on test) stays the target. The test figure meets it with only 1–2 questions
  of margin, so it is not locked yet.
- **The gate uses Recall@5 hit counts, not MRR.** Repeated runs give identical hits, but
  MRR moves in the third decimal (0.622 vs 0.630), so MRR would make the gate flaky.
- **A change in the scored question count fails the gate.** If the dataset changes, the
  baseline must be re-recorded (in its own PR, §13).
- **CI caches a Qdrant snapshot.** The cache key covers the corpus pin and all
  index-shaping code. A PR that touches ingest or retrieval code rebuilds the index on
  CPU, which is slow. Upgrade path: publish a prebuilt snapshot per corpus version.
- **Gate demonstrated locally.** `MEDLEAF_PREFETCH_LIMIT=3` (weaker hybrid prefetch) →
  `GATE FAILED: Recall@5 dropped by 2 questions (48 -> 46 of 57)`, exit 1. The default
  settings pass with exit 0. S8 still needs this shown in a real PR.
- **Skipped for now:** MLflow logging and `configs/best.yaml`. There is one retrieval
  configuration and MLflow is not a dependency yet. Add both when M5's reranker gives a
  second configuration to compare.

## 2026-10-16 — M4 slice 1: cited answers

- **The LLM is called over the OpenAI-compatible chat API with `httpx`, not LiteLLM**
  (SPEC §7). OpenRouter, OpenAI and Ollama all accept the same `/chat/completions`
  stream, so one 40-line function keeps the provider swappable through
  `MEDLEAF_LLM_BASE_URL` / `MEDLEAF_LLM_MODEL` without LiteLLM's large dependency tree.
  Revisit if a provider without that API is needed. Default: OpenRouter, model
  `openrouter/free`; the key is read from `OPENROUTER_API_KEY` or `MEDLEAF_LLM_API_KEY`.
- **The 30 s LLM timeout bounds each wait** (connect, and the gap between tokens), not
  the whole answer, so a long answer that keeps streaming is not cut off.
- **Invented citations are removed before they reach the client.** Text after `[` is
  held until `]` arrives and the key is checked against the six sources; unknown keys
  are dropped and listed in `done.invented_citations` and the log.
- **`/v1/ask` searches both languages** (top 20, no filters) and keeps six distinct
  chunks, preferring the question language's version of a chunk found in both.
- **Not in slice 1:** the safety check (no `safety` event yet), medicine-name filtering,
  the cross-encoder reranker and the score-threshold refusal. Until then `done.refused`
  means "the answer has no valid citation".

## 2026-10-16 — M4 slice 2A: safety check and medicine filter

- **Order (SPEC §9):** emergency keyword rules run first and skip the LLM, so an
  emergency never waits on a slow or failing model. Otherwise the LLM labels the
  question, and a personal keyword match raises an `info` label to `personal`. An LLM
  error or an unreadable reply counts as `personal`. The label and its source
  (`keywords`, `llm`, `fallback`) are logged for every question.
- **Rules need a first-person or present-tense signal** ("I took too many", "my son
  swallowed", "θέλω να πεθάνω"), so "What are the symptoms of overdose?" or "Does Lyrica
  cause suicidal thoughts?" still reach the documents. Text is lower-cased and
  accent-stripped first; `casefold` turns `ς` into `σ`, so the patterns use `σ`.
- **The emergency message always lists 1018 (Klimaka)**, as well as 112, 166 and the Poison
  Information Centre. Not only for suicidal wording, because keyword rules cannot
  reliably separate intent from accident. 1018 confirmed as Klimaka's 24-hour
  suicide-intervention line, 2026-10-16.
- **Greek substance names live in `retrieval/names.py`, not `medicines.yaml`.** That
  file's hash is the corpus version, so editing it would invalidate the index and the
  CI cache. The names are stems (Greek inflects the ending) taken from each Greek SmPC
  section 2. The 11 medicines whose Greek documents write the substance in Latin script
  need no entry. Brands come from the slugs, English substances from the manifest.
- **Known issue:** the `openrouter/free` router sometimes picks a guard model, which
  answers "User Safety: unsafe" instead of an answer (seen once in the live check). Pin a
  specific model through `MEDLEAF_LLM_MODEL` if it recurs.
