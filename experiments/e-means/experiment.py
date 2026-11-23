"""k-means vs e-means on the E5 chunk vectors, scored with the project's Recall@5.

Usage: python experiment.py KAGGLE_OUTPUT_DIR
Purity (what each cluster is made of) is reported for medicine, document kind, section and
language; it is not a retrieval score, and it rises with k, so compare the methods at the
same k only.
Scoring is dense-only (no sparse channel, no RRF), so its Recall@5 is not comparable with
eval/baseline.json; it compares the clustering methods with each other.
The settings below are fixed before seeing results (see PLAN.md); do not tune them
against the output.

e-means (arXiv:2611.09501), simplified: per-cluster count and sum are exponential moving
averages over random mini-batches; centroid = sum / count. No step annealing, no
dead-cluster revival, k-means++ start for both methods.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from medleaf.eval.metrics import gold_by_medicine, hit_rank, score

SEEDS = (0, 1, 2)
NPROBES = (1, 4, 8)          # clusters searched per query
LLOYD_ITERS = 25
EMEANS_EPOCHS, EMEANS_BATCH, EMEANS_DECAY = 2, 256, 0.99


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


def emeans(x, k, rng):
    c = kmeanspp(x, k, rng)
    count, total = np.full(k, 1e-10), c * 1e-10
    for _ in range(int(EMEANS_EPOCHS * len(x) / EMEANS_BATCH)):
        batch = x[rng.integers(0, len(x), EMEANS_BATCH)]
        labels, _ = nearest(batch, total / count[:, None])
        n_b = np.bincount(labels, minlength=k).astype(float)
        s_b = np.zeros_like(c)
        np.add.at(s_b, labels, batch)
        count = EMEANS_DECAY * count + (1 - EMEANS_DECAY) * n_b
        total = EMEANS_DECAY * total + (1 - EMEANS_DECAY) * s_b
    return total / count[:, None]


def ranked_ids(scores, chunk_ids, limit=10):
    """Chunk IDs of the best-scoring points. As in `medleaf eval`, the EN and EL copy of a
    section are separate hits, so both can take a place in the top k."""
    return [chunk_ids[i] for i in np.argsort(-scores)[:limit]]


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
    print("purity if every chunk were put in the largest label's group:", majority)

    rows = []
    for name, fit in (("kmeans", kmeans), ("emeans", emeans)):
        for seed in SEEDS:
            started = time.perf_counter()
            centers = fit(vectors, k, np.random.default_rng(seed))
            seconds = time.perf_counter() - started
            labels, dist = nearest(vectors, centers)
            row = {"method": name, "seed": seed, "seconds": round(seconds, 1),
                   "mse": round(float(dist.mean()), 4)}
            for name_, values in labels_of.items():
                row[f"purity_{name_}"] = purity(labels, values)
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
              "largest_label_share": majority, "rows": rows}
    (folder / "results.json").write_text(json.dumps(result, indent=1))
    print("wrote", folder / "results.json")


if __name__ == "__main__":
    main(sys.argv[1])
