# k-means vs e-means on the medicine embeddings

Experiment only. It does not touch the production index.

**Question.** If the E5 chunk vectors are split into clusters and each query searches only
its nearest clusters, does clustering with e-means (arXiv:2611.09501) find the right chunks
as often as k-means, and is it faster or tighter?

## Method

1. **Embed on Kaggle** (`kaggle/full-corpus/embed_corpus.py`, T4 GPU). It downloads the
   pinned EMA PDFs, chunks them with the project's own parser, and embeds them with the
   pinned E5 model, then embeds the dev questions.
   - 28 of the 80 PDFs (14 medicines) no longer match their pinned checksums, so EMA has
     published newer versions. Those medicines, and the dev questions that name them, are
     left out and listed in `manifest.json`. Expect about 26 medicines and 43 of 63 dev
     questions.
   - The held-out test questions are never embedded.
2. **Compare locally** (`experiment.py`): cluster the vectors with k-means and e-means
   (3 seeds each), route each question to its nearest 1, 4 or 8 clusters, and score with
   the project's own `Recall@5` (`medleaf.eval.metrics`). The exhaustive search over all
   chunks is the reference.

## Settings (fixed before seeing results)

- k = square root of the chunk count. Both methods use the same k and seeds.
- k-means: k-means++ start, 25 passes. e-means: k-means++ start, 2 passes over the data in
  batches of 256, decay 0.99. This is a simplified e-means: no step annealing and no
  dead-cluster revival (the paper has both).

## Decision

e-means is worth pursuing only if, at every cluster count searched, its mean Recall@5 hits
are within one question of k-means and it is faster or has lower error. If not, keep
k-means. With only about 43 questions, one question is about 2.3 percentage points, so a
small difference is noise; the results file shows all seeds.

## Run

```
# on Kaggle (from kaggle/full-corpus/), then download the output
kaggle kernels push -p . --accelerator NvidiaTeslaT4
kaggle kernels output nikoslitharis/medleaf-embed-corpus -p ../../out
# locally
python experiment.py out        # from experiments/e-means; writes out/results.json
```

The kernel metadata pins the GPU type and Docker image because the kernel stayed queued
without them. Which setting mattered was not isolated; keep all of them.

## Results so far

Kaggle GPU smoke test: 301 chunks embedded in 8 s on a T4 (about 28 times faster than
Kaggle CPU), norm error 1.2e-7. The full run stopped at the checksum check, which led to
the left-out medicines above. No clustering results yet.
