"""Retrieval service shared by the CLI and the HTTP API.

Both surfaces go through the same object so the acceptance ``curl`` call cannot succeed
by accident where ``medleaf search`` fails. Failures are typed:

* :class:`~medleaf.retrieval.index.IndexMissingError` — no collection / manifest / points.
* :class:`~medleaf.retrieval.index.RetrievalUnavailableError` — Qdrant unreachable or the
  embedding model could not be loaded.

The API maps both to HTTP 503 with a machine-readable code; nothing returns an empty
``results`` list to hide a broken index.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient, models

from medleaf.config import Settings
from medleaf.retrieval.embed import E5Encoder, Encoder, EncoderError
from medleaf.retrieval.index import (
    IndexManifest,
    IndexMissingError,
    RetrievalUnavailableError,
    build_filter,
    collection_count,
    corpus_version_for,
    hybrid_search,
    load_manifest,
    require_index,
)
from medleaf.retrieval.text import sparse_vector


class SearchHit(BaseModel):
    """One retrieved chunk with its full source provenance."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    score: float
    point_id: str
    chunk_id: str
    medicine: str
    active_substance: str | None = None
    atc_code: str | None = None
    lang: str
    doc: str
    section: str
    title: str
    text: str
    source_url: str
    revision_date: str | None = None
    corpus_version: str | None = None
    part: int = 1
    parts_total: int = 1


class SearchResponse(BaseModel):
    """Response shape of ``POST /v1/search`` and ``medleaf search``."""

    model_config = ConfigDict(extra="forbid")

    query: str
    lang: str | None = None
    medicine: list[str] | None = None
    doc: str | None = None
    active_substance: str | None = None
    corpus_version: str
    fusion: str = "rrf"
    prefetch_limit: int
    limit: int
    count: int
    latency_ms: float
    embed_model: str
    embed_revision: str
    results: list[SearchHit] = Field(default_factory=list)


class ReadyState(BaseModel):
    """Readiness probe result; ``detail`` explains a not-ready state."""

    model_config = ConfigDict(extra="forbid")

    ready: bool
    detail: str
    collection: str | None = None
    points: int | None = None
    corpus_version: str | None = None
    embed_model: str | None = None
    embed_revision: str | None = None


class RetrievalService:
    """Lazily-constructed hybrid retriever over the built Qdrant index."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: QdrantClient | None = None,
        encoder: Encoder | None = None,
    ) -> None:
        self.settings = settings
        self._client = client
        self._encoder = encoder

    @property
    def client(self) -> QdrantClient:
        """Qdrant client, created on first use and reused afterwards."""
        if self._client is None:
            try:
                self._client = QdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key,
                    timeout=int(self.settings.request_timeout_s),
                )
            except Exception as error:  # pragma: no cover - driver-specific
                msg = f"cannot create a Qdrant client for {self.settings.qdrant_url}: {error}"
                raise RetrievalUnavailableError(msg) from error
        return self._client

    @property
    def encoder(self) -> Encoder:
        """Embedding model, loaded on first use."""
        if self._encoder is None:
            self._encoder = E5Encoder.load(
                model_id=self.settings.embed_model,
                revision=self.settings.embed_revision,
                device=self.settings.embed_device,
                batch_size=self.settings.embed_batch_size,
                max_seq_length=self.settings.embed_max_seq_length,
                cache_dir=self.settings.embed_cache_dir,
            )
        return self._encoder

    @property
    def collection(self) -> str:
        return self.settings.qdrant_collection

    def manifest(self) -> IndexManifest:
        """Read the index manifest written by the last build."""
        return load_manifest(self.settings.index_manifest_file)

    def ready(self) -> ReadyState:
        """Cheap readiness check: Qdrant reachable, collection populated, manifest present."""
        try:
            manifest = self.manifest()
        except IndexMissingError as error:
            return ReadyState(ready=False, detail=str(error))
        try:
            if not self.client.collection_exists(self.collection):
                return ReadyState(
                    ready=False,
                    detail=(
                        f"collection {self.collection!r} does not exist; "
                        "run `medleaf index build`"
                    ),
                )
            points = collection_count(self.client, collection=self.collection)
        except Exception as error:
            return ReadyState(ready=False, detail=f"qdrant unavailable: {error}")
        if points == 0:
            return ReadyState(
                ready=False,
                detail=f"collection {self.collection!r} is empty; run `medleaf index build`",
            )
        return ReadyState(
            ready=True,
            detail=f"{points} points indexed",
            collection=self.collection,
            points=points,
            corpus_version=manifest.corpus_version,
            embed_model=manifest.embed_model,
            embed_revision=manifest.embed_revision,
        )

    def search(
        self,
        query: str,
        *,
        lang: str | None = None,
        medicine: Sequence[str] | None = None,
        doc: str | None = None,
        active_substance: str | None = None,
        corpus_version: str | None = None,
        limit: int | None = None,
    ) -> SearchResponse:
        """Run hybrid dense+sparse retrieval with RRF fusion and exact metadata filters."""
        if not query.strip():
            msg = "query must not be empty"
            raise ValueError(msg)
        manifest = self.manifest()
        version = corpus_version or manifest.corpus_version
        request_limit = limit or self.settings.default_limit
        query_filter = build_filter(
            medicine=medicine,
            lang=lang,
            doc=doc,
            corpus_version=version,
            active_substance=active_substance,
        )

        started = time.perf_counter()
        try:
            require_index(self.client, collection=self.collection, corpus_version=version)
            encoder = self.encoder
            dense = encoder.encode_queries([query])[0]
            sparse = sparse_vector(query)
            hits = hybrid_search(
                self.client,
                collection=self.collection,
                dense_vector=dense,
                sparse_query=sparse,
                dense_vector_name=self.settings.dense_vector_name,
                sparse_vector_name=self.settings.sparse_vector_name,
                limit=request_limit,
                prefetch_limit=self.settings.prefetch_limit,
                query_filter=query_filter,
            )
        except (IndexMissingError, ValueError):
            raise
        except EncoderError as error:
            raise RetrievalUnavailableError(str(error)) from error
        except Exception as error:
            msg = f"retrieval failed against {self.settings.qdrant_url}: {error}"
            raise RetrievalUnavailableError(msg) from error
        latency_ms = (time.perf_counter() - started) * 1000

        return SearchResponse(
            query=query,
            lang=lang,
            medicine=list(medicine) if medicine else None,
            doc=doc,
            active_substance=active_substance,
            corpus_version=version,
            prefetch_limit=self.settings.prefetch_limit,
            limit=request_limit,
            count=len(hits),
            latency_ms=round(latency_ms, 2),
            embed_model=encoder.model_id,
            embed_revision=encoder.revision,
            results=[
                _hit_from_point(point, rank=rank)
                for rank, point in enumerate(hits, start=1)
            ],
        )

    def default_corpus_version(self) -> str:
        """Corpus version used when a caller passes none: the indexed one, else the file's."""
        try:
            return self.manifest().corpus_version
        except IndexMissingError:
            return corpus_version_for(self.settings.medicines_file)


def _hit_from_point(point: models.ScoredPoint, *, rank: int) -> SearchHit:
    payload = point.payload or {}
    return SearchHit(
        rank=rank,
        score=float(point.score),
        point_id=str(point.id),
        chunk_id=str(payload.get("chunk_id", "")),
        medicine=str(payload.get("medicine", "")),
        active_substance=_optional_str(payload.get("active_substance")),
        atc_code=_optional_str(payload.get("atc_code")),
        lang=str(payload.get("lang", "")),
        doc=str(payload.get("doc", "")),
        section=str(payload.get("section", "")),
        title=str(payload.get("title", "")),
        text=str(payload.get("text", "")),
        source_url=str(payload.get("source_url", "")),
        revision_date=_optional_str(payload.get("revision_date")),
        corpus_version=_optional_str(payload.get("corpus_version")),
        part=_optional_int(payload.get("part"), default=1),
        parts_total=_optional_int(payload.get("parts_total"), default=1),
    )


def _optional_int(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value
    return default


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


__all__ = ["ReadyState", "RetrievalService", "SearchHit", "SearchResponse"]
