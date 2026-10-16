# Greek stemming

## Scope

Fix the Greek sparse-retrieval miss recorded in `feature-local-demo.md`. Change only
sparse tokenisation and index readiness. Dense model, prompts, corpus and the
held-out test split are unchanged.

## Changes

- `stem()` strips one folded Greek inflectional ending (longest first, stem of at
  least three letters). Latin tokens and digits are unchanged. Index and query share
  the rule. Stopwords are matched before stemming.
- `TOKENIZER_VERSION` is now `medleaf-tokenizer-2`.
- `/readyz` and search refuse an index built with another tokenizer version. Before,
  a mismatch did not error; query tokens silently stopped matching.

## Evidence

- Cause: `ναυτίας`/`Ναυτία`, `σεμαγλουτίδης`/`σεμαγλουτίδη` and `δόσεις`/`δόση` were
  distinct sparse tokens. The Greek nausea-rate question ranked the expected
  `ozempic:smpc:4.8#4` chunk below 100 on the sparse channel.
- Offline: 165 passed, 40 skipped; Ruff passed. Removing the tokenizer check made its
  new test fail.
- Full rebuild: 8,797 chunks, 75 minutes on CPU. The served name is now an alias.
- Dev gate before/after: Recall@5 48/57 both (Greek 24/30, English 24/27).
  MRR@10 0.618 -> 0.674 (Greek 0.600 -> 0.650, English 0.638 -> 0.700).
  English tokens are unchanged, so English movement is rebuild noise from the new
  approximate (HNSW) graph; the Greek MRR gain is not cleanly attributable either.
- Ozempic (train split), expected chunk rank with the medicine filter:

  | Question | Sparse before/after | Hybrid before/after |
  |---|---|---|
  | Greek nausea rates | >100 / 13 | 20 / 11 |
  | Greek "19.9% at 0.5 mg?" | 11 / 9 | 26 / 22 |
  | Greek general side effects | - | 4 / 18 |
  | English nausea rates | 4 / 4 | 5 / 5 |

- The running API container received the two changed files by `docker cp` and a
  restart, preserving its NVIDIA environment. Installed hashes match the working
  tree. Live search reproduces rank 11. The image itself is not rebuilt.

## Result

Not accepted. Stemming fixed the token mismatch, but no dev recall gain and the
expected chunk is still outside the six chunks passed to Ask. The general Greek
side-effect question moved this chunk down.

## Remaining cause

Sparse values are raw term frequency with no saturation or length normalisation.
`ozempic:smpc:2#1` (composition) contains `σεμαγλουτιδ` 20 times and wins every
Ozempic sparse query; the target contains `ναυτι` once. Candidate next step:
BM25-style saturated term frequency at index time. Only sparse vectors change, so
Qdrant `update_vectors` could avoid re-embedding. Evaluate on the dev split only.
