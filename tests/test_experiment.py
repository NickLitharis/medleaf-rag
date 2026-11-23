"""The e-means comparison script: scoring must not count chunks that were not searched."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "e-means"))

import experiment as exp  # noqa: E402


def _unit(rows: int, dim: int = 8, seed: int = 0):
    vectors = np.random.default_rng(seed).normal(size=(rows, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_chunks_outside_the_searched_clusters_never_count_as_hits() -> None:
    vectors = _unit(20)
    ids = [f"m{i}:smpc:4.1" for i in range(20)]
    question = [{"gold_chunk_ids": ["m0:smpc:4.1"], "medicines": ["m0"]}]
    far_from_gold = -vectors[:1]
    # Only rows 10 and 11 are searched, so the gold chunk (row 0) cannot be found.
    hits = exp.recall_at_5(question, far_from_gold, vectors, ids,
                           allowed=lambda _n: np.array([10, 11]))
    assert hits == 0


def test_searching_all_clusters_finds_the_gold_chunk() -> None:
    vectors = _unit(20)
    ids = [f"m{i}:smpc:4.1" for i in range(20)]
    question = [{"gold_chunk_ids": ["m3:smpc:4.1"], "medicines": ["m3"]}]
    assert exp.recall_at_5(question, vectors[3:4], vectors, ids) == 1


def test_purity_is_one_for_pure_clusters_and_lower_for_mixed_ones() -> None:
    clusters = np.array([0, 0, 1, 1])
    assert exp.purity(clusters, ["a", "a", "b", "b"]) == 1.0
    assert exp.purity(clusters, ["a", "b", "a", "b"]) == 0.5


def test_emeans_fits_separated_blobs_about_as_tightly_as_kmeans() -> None:
    rng = np.random.default_rng(1)
    centres = rng.normal(size=(6, 16)) * 4
    x = (centres[rng.integers(6, size=1200)] + rng.normal(size=(1200, 16))).astype(np.float32)
    k = 6
    lloyd = exp.nearest(x, exp.kmeans(x, k, np.random.default_rng(0)))[1].mean()
    moving = exp.nearest(x, exp.emeans(x, k, np.random.default_rng(0), 64, 0.2))[1].mean()
    assert moving < lloyd * 1.1


def test_revival_gives_an_empty_cluster_a_share_of_a_busy_one() -> None:
    count = np.array([10.0, 10.0, 0.0])
    total = np.array([[10.0, 0.0], [0.0, 10.0], [0.0, 0.0]])
    exp.revive(count, total, np.random.default_rng(0))
    assert count[2] > 0
    assert count.sum() == pytest.approx(20.0)
