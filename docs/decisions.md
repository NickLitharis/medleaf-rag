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
