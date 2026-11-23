# k-means vs e-means on the medicine embeddings

Experiment only. It does not touch the production index.

**Question.** If the E5 chunk vectors are split into clusters and each query searches only
its nearest clusters, does clustering with e-means (arXiv:2611.09501) find the right chunks
as often as k-means, and is it faster or tighter?

## Method

1. **Download once, then embed, on Kaggle.** `kaggle/ema-pdfs` downloads the pinned EMA PDFs
   and checks their checksums (CPU). `kaggle/full-corpus/embed_corpus.py` (T4 GPU) reads
   those PDFs, chunks them with the project's own parser, embeds them with the pinned E5
   model, and embeds the dev questions.
   - 28 of the 80 PDFs (14 medicines) no longer match their pinned checksums, so EMA has
     published newer versions. Those medicines, and the dev questions that name them, are
     left out and listed in `manifest.json`. Expect about 26 medicines and 43 of 63 dev
     questions.
   - The held-out test questions are never embedded.
2. **Compare locally** (`experiment.py`): cluster the vectors with k-means and e-means
   (3 seeds each), route each question to its nearest 1, 2 or 4 clusters, and score with
   the project's own `Recall@5` (`medleaf.eval.metrics`). The exhaustive search over all
   chunks is the reference.
   It also reports cluster purity: the share of chunks that carry their cluster's most
   common label, for medicine, document kind (SmPC or leaflet), section and language. This
   shows what each method grouped together. Purity rises with k, so compare the methods at
   the same k, and read it against the share of the largest label overall (also reported).

## Settings (fixed before seeing results)

- k = square root of the chunk count. Both methods use the same k, the same k-means++ start
  and the same seeds (0, 1, 2).
- k-means: 25 passes.
- e-means follows Algorithm 1 of the paper: exponential moving averages of cluster count and
  sum over shuffled mini-batches, 3 passes over the data, a cosine-annealed step (down to
  10% of its peak), dead-cluster revival every 5 batches, and the last-epoch-mean final
  step. Batch size (32 to 512) and peak step (0.05 to 0.4) are picked by a small grid on
  training error with a separate seed. No question is used to pick them. The script warns if
  the best setting sits on the edge of the grid.
- Differences from the paper: a k-means++ start instead of a random one, and the grid instead
  of the paper's scaling laws (which the paper says are not tuned for small datasets).
- Questions score only the clusters they are routed to. A chunk that was not searched never
  counts as a hit, even when a searched cluster has fewer than 5 chunks.

## Decision

e-means is worth pursuing only if, at every cluster count searched, its mean Recall@5 hits
are within one question of k-means and it is faster or has lower error. If not, keep
k-means. With only about 43 questions, one question is about 2.3 percentage points, so a
small difference is noise; the results file shows all seeds.

## Run

Nothing is downloaded twice. The PDFs are fetched once, by their own small kernel, and the
embedding kernel reads that kernel's output. The vectors are pulled to `out/` once, so
clustering runs need neither Kaggle nor EMA.

```
cd experiments/e-means/kaggle
# 1. Once (and again only if data/medicines.yaml changes): download + verify the PDFs
kaggle kernels push -p ema-pdfs            # CPU; wait until it completes
# 2. Embed (GPU); reads kernel 1's output as an input
kaggle kernels push -p full-corpus --accelerator NvidiaTeslaT4
kaggle kernels output nikoslitharis/medleaf-embed-corpus -p ../out
# 3. Compare, as often as you like, offline
cd .. && python experiment.py out          # writes out/results.json
```

Re-run step 2 only if the parser, the model or the chunk budget changes. Re-run step 1
only if the corpus definition changes.

The GPU kernel's metadata pins the GPU type and Docker image because it stayed queued
without them. The CPU download kernel needs neither (a CPU kernel ran fine on the default image). Which setting mattered was not isolated; keep all of them.

## Results so far

Kaggle GPU smoke test: 301 chunks embedded in 8 s on a T4 (about 28 times faster than
Kaggle CPU), norm error 1.2e-7. The full run stopped at the checksum check, which led to
the left-out medicines above. No clustering results yet.
