"""k-means vs e-means on the E5 chunk vectors, scored with the project's Recall@5.

Usage: python experiment.py KAGGLE_OUTPUT_DIR
Settings are fixed in the constants below before any result is seen (see PLAN.md).
Scoring is dense-only (no sparse channel, no RRF), so its Recall@5 is not comparable with
any hybrid-search figure; it compares the two clusterings with each other.
Purity (what each cluster is made of) is reported for medicine, document kind, section and
language. It is not a retrieval score and it rises with k, so compare methods at the same k.

e-means follows Algorithm 1 of arXiv:2611.09501: exponential moving averages of per-cluster
count and sum over shuffled mini-batches, a cosine-annealed step, dead-cluster revival by
splitting a donor, and a last-epoch-mean finalisation. Differences from the paper: both
methods start from k-means++ (the paper uses random starts) and the batch size and peak
step are picked by a small grid on training error, not by the paper's scaling laws.
"""
import json
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from medleaf.eval.metrics import gold_by_medicine, hit_rank, score

SEEDS = (0, 1, 2)
TUNING_SEED = 100              # distinct from the reported seeds
NPROBES = (1, 2, 4)            # clusters searched per question
LLOYD_ITERS = 25
EMEANS_EPOCHS = 3
BATCH_GRID = (32, 64, 128, 256, 512)
ALPHA_GRID = (0.05, 0.1, 0.2, 0.4)    # peak step 1 - decay, annealed to 10% of this
ALPHA_FLOOR = 0.1
REVIVE_EVERY = 5               # batches
REVIVE_BELOW = 0.1             # of the mean cluster count
SPLIT_EPS = 1 / 1024           # relative shift between a donor and the cluster it revives


def nearest(x, centers):
    """Index and squared distance of the closest center for every row of x."""
    d = (x**2).sum(1)[:, None] - 2 * x @ centers.T + (centers**2).sum(1)[None, :]
    return d.argmin(1), d.min(1)


def kmeanspp(x, k, rng):
    centers = [x[rng.integers(len(x))]]
    d2 = ((x - centers[0]) ** 2).sum(1)
    for _ in range(k - 1):
        centers.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((x - centers[-1]) ** 2).sum(1))
    return np.stack(centers)


def kmeans(x, k, rng):
    c = kmeanspp(x, k, rng)
    for _ in range(LLOYD_ITERS):
        labels, _ = nearest(x, c)
        for j in range(k):
            members = x[labels == j]
            if len(members):
                c[j] = members.mean(0)
    return c


def revive(count, total, rng):
    """Give clusters that have nearly stopped receiving points a share of a busy cluster."""
    dead = count < REVIVE_BELOW * count.mean()
    if not dead.any() or dead.all():
        return
    for j in np.flatnonzero(dead):
        weight = np.where(dead, 0.0, count)
        donor = rng.choice(len(count), p=weight / weight.sum())
        centroid = total[donor] / count[donor]
        count[donor] /= 2
        count[j] = count[donor]
        total[donor] = count[donor] * centroid * (1 - SPLIT_EPS)
        total[j] = count[j] * centroid * (1 + SPLIT_EPS)


def emeans(x, k, rng, batch, alpha0):
    n = len(x)
    c = kmeanspp(x, k, rng).astype(np.float64)
    count, total = np.full(k, 1e-10), c * 1e-10
    per_epoch = n // batch
    steps = EMEANS_EPOCHS * per_epoch
    t = 0
    for epoch in range(EMEANS_EPOCHS):
        order = rng.permutation(n)
        last_sum, last_size = np.zeros_like(c), np.zeros(k)
        for b in range(per_epoch):
            points = x[order[b * batch:(b + 1) * batch]]
            labels, _ = nearest(points, total / count[:, None])
            cosine = 0.5 * (1 + np.cos(np.pi * t / steps))
            alpha = alpha0 * (ALPHA_FLOOR + (1 - ALPHA_FLOOR) * cosine)
            sizes = np.bincount(labels, minlength=k).astype(float)
            sums = np.zeros_like(c)
            np.add.at(sums, labels, points)
            count = (1 - alpha) * count + alpha * sizes
            total = (1 - alpha) * total + alpha * sums
            t += 1
            if t % REVIVE_EVERY == 0:
                revive(count, total, rng)
            if epoch == EMEANS_EPOCHS - 1:
                last_sum += sums
                last_size += sizes
    centers = total / count[:, None]
    seen = last_size > 0
    centers[seen] = last_sum[seen] / last_size[seen, None]   # last-epoch mean
    return centers


def tune_emeans(x, k):
    """Pick batch size and peak step by lowest training error. No questions are used."""
    best = None
    for batch in BATCH_GRID:
        for alpha0 in ALPHA_GRID:
            centers = emeans(x, k, np.random.default_rng(TUNING_SEED), batch, alpha0)
            error = float(nearest(x, centers)[1].mean())
            if best is None or error < best[0]:
                best = (error, batch, alpha0)
    chosen = {"mse": round(best[0], 4), "batch": best[1], "alpha0": best[2]}
    edge = best[1] in (BATCH_GRID[0], BATCH_GRID[-1]) or best[2] in (ALPHA_GRID[0], ALPHA_GRID[-1])
    if edge:
        print("warning: the best e-means setting is on the edge of the grid; widen it")
    return chosen


def purity(cluster_of, labels):
    """Share of chunks that carry their cluster's most common label (1.0 = every cluster
    holds a single label). Compare it with the share of the largest label overall."""
    _, label_codes = np.unique(labels, return_inverse=True)
    counts = np.zeros((cluster_of.max() + 1, label_codes.max() + 1), dtype=int)
    np.add.at(counts, (cluster_of, label_codes), 1)
    return round(float(counts.max(1).sum() / len(labels)), 4)


def chunk_labels(rows):
    """What each chunk is: medicine, document kind, section across medicines, language."""
    parts = [row["chunk_id"].split("#")[0].split(":") for row in rows]
    return {
        "medicine": [p[0] for p in parts],
        "doc": [p[1] for p in parts],
        "section": [f"{p[1]}:{p[2]}" for p in parts],
        "lang": [row["lang"] for row in rows],
    }


def ranked_ids(scores, chunk_ids, limit=10):
    """Chunk IDs of the best-scoring points. As in `medleaf eval`, the EN and EL copy of a
    section are separate hits. Points that were not searched (score -inf) never count,
    even when the searched clusters hold fewer than `limit` chunks."""
    top = np.argsort(-scores)[:limit]
    return [chunk_ids[i] for i in top if np.isfinite(scores[i])]


def recall_at_5(queries, qvecs, vectors, chunk_ids, allowed=None):
    """Recall@5 hits; `allowed(q_index)` limits search to a subset of chunk rows."""
    ranks = []
    for n, q in enumerate(queries):
        rows = np.arange(len(vectors)) if allowed is None else allowed(n)
        scores = np.full(len(vectors), -np.inf)
        scores[rows] = vectors[rows] @ qvecs[n]
        ids = ranked_ids(scores, chunk_ids)
        ranks.append(hit_rank(ids, gold_by_medicine(q["gold_chunk_ids"])))
    return score(ranks)["recall_at_5"]["hits"]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main(folder):
    folder = Path(folder)
    vectors = np.load(folder / "vectors.npy")
    qvecs = np.load(folder / "query_vectors.npy")
    chunk_rows = read_jsonl(folder / "chunks.jsonl")
    chunk_ids = [row["chunk_id"] for row in chunk_rows]
    labels_of = chunk_labels(chunk_rows)
    majority = {name: round(max(Counter(v).values()) / len(v), 4) for name, v in labels_of.items()}
    queries = read_jsonl(folder / "queries.jsonl")
    k = round(len(vectors) ** 0.5)
    baseline = recall_at_5(queries, qvecs, vectors, chunk_ids)
    print(f"{len(vectors)} chunks, {len(queries)} questions, k={k}; "
          f"exhaustive Recall@5 = {baseline}/{len(queries)}")
    print("share of the largest label (purity floor):", majority)
    tuned = tune_emeans(vectors, k)
    print("e-means settings chosen by training error:", tuned)

    fits = (("kmeans", kmeans),
            ("emeans", partial(emeans, batch=tuned["batch"], alpha0=tuned["alpha0"])))
    rows = []
    for name, fit in fits:
        for seed in SEEDS:
            started = time.perf_counter()
            centers = fit(vectors, k, np.random.default_rng(seed))
            seconds = time.perf_counter() - started
            labels, dist = nearest(vectors, centers)
            row = {"method": name, "seed": seed, "seconds": round(seconds, 1),
                   "mse": round(float(dist.mean()), 4),
                   "smallest_cluster": int(np.bincount(labels, minlength=k).min())}
            for label_name, values in labels_of.items():
                row[f"purity_{label_name}"] = purity(labels, values)
            # Nearest clusters by L2: argmin |q-c|^2 = argmax q.c - |c|^2/2.
            affinity = qvecs @ centers.T - (centers**2).sum(1)[None, :] / 2
            for p in NPROBES:
                near = np.argsort(-affinity, axis=1)[:, :p]
                row[f"hits@nprobe{p}"] = recall_at_5(
                    queries, qvecs, vectors, chunk_ids,
                    allowed=lambda n, near=near, labels=labels: np.flatnonzero(
                        np.isin(labels, near[n])),
                )
            rows.append(row)
            print(row, flush=True)
    result = {"baseline": baseline, "questions": len(queries), "k": k,
              "largest_label_share": majority, "emeans_settings": tuned, "rows": rows}
    (folder / "results.json").write_text(json.dumps(result, indent=1))
    print("wrote", folder / "results.json")


if __name__ == "__main__":
    main(sys.argv[1])
