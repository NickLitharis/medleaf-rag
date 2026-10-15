"""Hybrid retrieval: e5 dense vectors plus Qdrant sparse IDF vectors (SPEC section 9).

The pieces are deliberately separable so each contract can be tested on its own:

* :mod:`medleaf.retrieval.text` — accent-stripped tokenisation and sparse encoding.
* :mod:`medleaf.retrieval.embed` — the ``passage:`` / ``query:`` e5 contract.
* :mod:`medleaf.retrieval.index` — Qdrant collection, point IDs, filters, RRF search.
* :mod:`medleaf.retrieval.pipeline` — parse cache and index build.
"""

from __future__ import annotations

__all__: list[str] = []
