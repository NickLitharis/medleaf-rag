"""Qdrant index for hybrid retrieval: named dense vectors plus sparse IDF vectors.

Design decisions worth stating, because they are the acceptance-relevant ones:

* **Logical vs physical IDs.** The chunk ID (``eliquis:smpc:4.5``) is language
  independent and shared by the Greek and English text of the same section. Qdrant
  point IDs must be unique, so each language gets its own point:
  ``uuid5(NAMESPACE, "<lang>:<chunk_id>")``. Same chunk ID, distinct physical points.
* **Sparse IDF.** The ``sparse`` vector is created with ``Modifier.IDF``, so Qdrant
  computes inverse document frequency over the collection and weights the sparse
  scores itself. No side vocabulary or IDF file can drift from the collection.
* **Fusion.** Dense and sparse candidates are each retrieved with their own prefetch
  (top :data:`DEFAULT_PREFETCH_LIMIT`) and merged by Qdrant's reciprocal rank fusion
  (``Fusion.RRF``) in a single Query API call — the fusion runs in Qdrant, not in a
  local reimplementation.
* **Filters.** ``medicine``, ``lang``, ``doc`` and ``corpus_version`` are exact
  payload filters applied to both prefetches and the fused query.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient, models

from medleaf.models import Medicine, Section
from medleaf.retrieval.embed import Encoder
from medleaf.retrieval.text import TOKENIZER_VERSION, sparse_vector

#: Fixed namespace for chunk point IDs; changing it invalidates every built index.
POINT_NAMESPACE = uuid.UUID("1a2f6e2c-9c0f-4d5b-9f1c-7f1e4c2b8a10")

#: Candidates taken from each of the dense and sparse prefetches before RRF fusion.
DEFAULT_PREFETCH_LIMIT = 30

#: Index manifest schema; bump when the meaning of a field changes.
MANIFEST_SCHEMA_VERSION = 1


class IndexMissingError(RuntimeError):
    """The index does not exist, is empty, or its manifest cannot be read."""


class IndexIncompleteError(IndexMissingError):
    """The collection holds a different number of points than the manifest records.

    A full build drops the collection first and writes the manifest last, so a build
    that is still running (or that died) leaves a partial collection next to the
    previous build's manifest. Serving it would answer from part of the corpus.
    """


class RetrievalUnavailableError(RuntimeError):
    """The retrieval service (Qdrant) cannot be reached."""


class IndexManifest(BaseModel):
    """Everything needed to know what a built index contains (SPEC section 15)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = MANIFEST_SCHEMA_VERSION
    collection: str
    corpus_version: str
    corpus_sha256: str = Field(description="SHA-256 of the medicines.yaml that was indexed.")
    embed_model: str
    embed_revision: str
    dense_dim: int
    dense_vector_name: str
    sparse_vector_name: str
    sparse_modifier: str = "idf"
    tokenizer: str = TOKENIZER_VERSION
    built_at: str
    total_chunks: int
    chunk_counts: dict[str, int] = Field(default_factory=dict)
    medicines: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    missing: list[str] = Field(
        default_factory=list,
        description=(
            "Requested medicine/language pairs with no parsed source; reported, never indexed."
        ),
    )
    active_substances: dict[str, str | None] = Field(
        default_factory=dict,
        description="Per-medicine substance name from the corpus manifest (SPEC section 8).",
    )
    atc_codes: dict[str, str | None] = Field(
        default_factory=dict,
        description="Per-medicine ATC code from the corpus manifest (SPEC section 8).",
    )
    missing_metadata: list[str] = Field(
        default_factory=list,
        description=(
            "'<slug>:<field>' entries for manifest metadata EMA did not supply; recorded "
            "as unavailable rather than invented."
        ),
    )
    split_sections: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "'<slug>:<lang>:<doc>:<section>' -> part count for sections longer than one "
            "embedding window; these are the chunks whose parts are not language-shared."
        ),
    )

    @property
    def is_complete(self) -> bool:
        """True when every requested language had chunks and nothing was skipped."""
        return not self.missing and self.total_chunks > 0


def corpus_sha256(medicines_file: Path) -> str:
    """Hash the corpus manifest; used as the default ``corpus_version``."""
    return hashlib.sha256(medicines_file.read_bytes()).hexdigest()


def corpus_version_for(medicines_file: Path) -> str:
    """Human-readable corpus version derived from the manifest contents."""
    return f"medicines.yaml:{corpus_sha256(medicines_file)[:12]}"


def point_id(chunk_id: str, lang: str) -> str:
    """Physical Qdrant point ID: stable per ``(lang, chunk_id)`` and unique per language."""
    return str(uuid.uuid5(POINT_NAMESPACE, f"{lang}:{chunk_id}"))


def build_filter(
    *,
    medicine: Sequence[str] | None = None,
    lang: str | None = None,
    doc: str | None = None,
    corpus_version: str | None = None,
    active_substance: str | None = None,
) -> models.Filter | None:
    """Build an exact-match filter over the indexed payload fields."""
    conditions: list[models.Condition] = []
    if medicine:
        conditions.append(
            models.FieldCondition(key="medicine", match=models.MatchAny(any=list(medicine)))
        )
    if lang:
        conditions.append(models.FieldCondition(key="lang", match=models.MatchValue(value=lang)))
    if doc:
        conditions.append(models.FieldCondition(key="doc", match=models.MatchValue(value=doc)))
    if corpus_version:
        conditions.append(
            models.FieldCondition(
                key="corpus_version", match=models.MatchValue(value=corpus_version)
            )
        )
    if active_substance:
        conditions.append(
            models.FieldCondition(
                key="active_substance", match=models.MatchValue(value=active_substance)
            )
        )
    return models.Filter(must=conditions) if conditions else None


def medicine_metadata(medicine: Medicine) -> dict[str, object]:
    """Chunk metadata taken from the resolved corpus manifest (SPEC section 8).

    A field EMA did not publish is stored as ``None`` and listed in the manifest's
    ``missing_metadata``; it is reported as unavailable, never invented.
    """
    return {
        "active_substance": medicine.active_substance or None,
        "atc_code": medicine.atc_code or None,
    }


def missing_metadata(medicines: Sequence[Medicine]) -> list[str]:
    """'<slug>:<field>' entries for manifest metadata that is genuinely absent."""
    missing: list[str] = []
    for medicine in medicines:
        if not medicine.active_substance:
            missing.append(f"{medicine.slug}:active_substance")
        if not medicine.atc_code:
            missing.append(f"{medicine.slug}:atc_code")
    return missing


def payload_for(
    section: Section,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Payload carried by every point; the API returns it as citation provenance."""
    payload: dict[str, object] = {
        "chunk_id": section.chunk_id,
        "medicine": section.medicine,
        "lang": section.lang.value,
        "doc": section.doc.value,
        "section": section.section,
        "title": section.title,
        "text": section.text,
        "source_url": section.source_url,
        "revision_date": section.revision_date,
        "corpus_version": section.corpus_version,
        "part": section.part,
        "parts_total": section.parts_total,
    }
    if metadata:
        payload.update(metadata)
    return payload


def ensure_collection(
    client: QdrantClient,
    *,
    collection: str,
    dense_dim: int,
    dense_vector_name: str,
    sparse_vector_name: str,
    recreate: bool = False,
) -> bool:
    """Create the collection when needed; return True when it was (re)created."""
    exists = client.collection_exists(collection)
    if exists and not recreate:
        return False
    if exists:
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config={
            dense_vector_name: models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            sparse_vector_name: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
                modifier=models.Modifier.IDF,
            )
        },
    )
    return True


def build_collection_name(served: str, now: datetime | None = None) -> str:
    """Physical collection a full build fills before ``served`` is pointed at it."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{served}_{stamp}"


def alias_target(client: QdrantClient, alias: str) -> str | None:
    """Physical collection behind ``alias``, or ``None`` when no such alias exists."""
    for entry in client.get_aliases().aliases:
        if entry.alias_name == alias:
            return entry.collection_name
    return None


def point_alias(client: QdrantClient, *, alias: str, collection: str) -> str | None:
    """Atomically point ``alias`` at ``collection``; return the collection it left, if any.

    Readers keep using the alias name throughout, so they see the old index until the
    switch and the new one after it — never a half-built collection. A real collection
    still carrying the alias name (the pre-alias layout) is deleted first, because Qdrant
    refuses an alias that shadows a collection; that one-time migration is the only gap.
    """
    previous = alias_target(client, alias)
    operations: list[models.AliasOperations] = []
    if previous is not None:
        operations.append(
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
        )
    elif client.collection_exists(alias):
        client.delete_collection(alias)
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)
    return previous if previous != collection else None


def upsert_sections(
    client: QdrantClient,
    *,
    collection: str,
    sections: Sequence[Section],
    dense_vectors: Sequence[Sequence[float]],
    dense_vector_name: str,
    sparse_vector_name: str,
    metadata: Mapping[str, Mapping[str, object]] | None = None,
    wait: bool = True,
) -> int:
    """Upsert sections with their dense and sparse vectors; return the point count.

    ``metadata`` is keyed by medicine slug and merged into each point's payload, so
    manifest metadata (SPEC section 8) travels with the chunk.
    """
    if len(sections) != len(dense_vectors):
        msg = f"{len(sections)} sections but {len(dense_vectors)} dense vectors"
        raise ValueError(msg)
    points = [
        models.PointStruct(
            id=point_id(section.chunk_id, section.lang.value),
            vector={
                dense_vector_name: list(vector),
                sparse_vector_name: sparse_vector(section.text),
            },
            payload=payload_for(
                section, None if metadata is None else metadata.get(section.medicine)
            ),
        )
        for section, vector in zip(sections, dense_vectors, strict=True)
    ]
    if points:
        client.upsert(collection_name=collection, points=points, wait=wait)
    return len(points)


def _medicine_selector(slug: str) -> models.Filter:
    """Filter selecting every point of one medicine (both languages, all documents)."""
    return models.Filter(
        must=[models.FieldCondition(key="medicine", match=models.MatchValue(value=slug))]
    )


def set_medicine_metadata(
    client: QdrantClient,
    *,
    collection: str,
    metadata_by_slug: Mapping[str, Mapping[str, object]],
) -> dict[str, int]:
    """Attach payload metadata to existing points without re-embedding anything.

    Returns the number of points updated per medicine slug. Enriching an already built
    index (for example after a manifest-metadata fix) costs a payload update, not a
    re-embed, and the vectors are left untouched.
    """
    updated: dict[str, int] = {}
    for slug, metadata in sorted(metadata_by_slug.items()):
        query_filter = _medicine_selector(slug)
        points = collection_count(client, collection=collection, query_filter=query_filter)
        client.set_payload(
            collection_name=collection,
            payload=dict(metadata),
            points=models.FilterSelector(filter=query_filter),
            wait=True,
        )
        updated[slug] = points
    return updated


def delete_medicine_points(client: QdrantClient, *, collection: str, slug: str) -> int:
    """Delete every point of one medicine; returns the number removed.

    Used by a targeted rebuild so stale chunk IDs from an older parse cannot survive.
    """
    query_filter = _medicine_selector(slug)
    points = collection_count(client, collection=collection, query_filter=query_filter)
    if points:
        client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(filter=query_filter),
            wait=True,
        )
    return points


def hybrid_search(
    client: QdrantClient,
    *,
    collection: str,
    dense_vector: Sequence[float],
    sparse_query: models.SparseVector,
    dense_vector_name: str,
    sparse_vector_name: str,
    limit: int,
    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
    query_filter: models.Filter | None = None,
) -> list[models.ScoredPoint]:
    """Run dense and sparse retrieval and fuse the candidates with Qdrant RRF."""
    prefetch = [
        models.Prefetch(
            query=list(dense_vector),
            using=dense_vector_name,
            limit=prefetch_limit,
            filter=query_filter,
        ),
        models.Prefetch(
            query=sparse_query,
            using=sparse_vector_name,
            limit=prefetch_limit,
            filter=query_filter,
        ),
    ]
    result = client.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )
    return list(result.points)


def collection_count(
    client: QdrantClient,
    *,
    collection: str,
    query_filter: models.Filter | None = None,
) -> int:
    """Exact point count, optionally filtered."""
    return int(
        client.count(collection_name=collection, count_filter=query_filter, exact=True).count
    )


def require_index(
    client: QdrantClient,
    *,
    collection: str,
    corpus_version: str | None,
    expected_points: int | None = None,
) -> None:
    """Raise :class:`IndexMissingError` unless a non-empty index is present.

    With ``expected_points`` (the manifest's ``total_chunks``), a collection holding any
    other number of points raises :class:`IndexIncompleteError`.
    """
    if not client.collection_exists(collection):
        msg = f"collection {collection!r} does not exist; run `medleaf index build`"
        raise IndexMissingError(msg)
    if expected_points is not None:
        check_complete(client, collection=collection, expected_points=expected_points)
    query_filter = build_filter(corpus_version=corpus_version)
    if collection_count(client, collection=collection, query_filter=query_filter) == 0:
        detail = f" (corpus_version={corpus_version!r})" if corpus_version else ""
        msg = f"collection {collection!r} has no points{detail}; run `medleaf index build`"
        raise IndexMissingError(msg)


def check_complete(client: QdrantClient, *, collection: str, expected_points: int) -> int:
    """Return the point count, raising :class:`IndexIncompleteError` on a mismatch."""
    points = collection_count(client, collection=collection)
    if points != expected_points:
        msg = (
            f"collection {collection!r} has {points} points but the manifest records "
            f"{expected_points}; an index build is running or did not finish"
        )
        raise IndexIncompleteError(msg)
    return points


def write_manifest(path: Path, manifest: IndexManifest) -> None:
    """Persist the index manifest as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> IndexManifest:
    """Read the index manifest, raising :class:`IndexMissingError` when absent or broken."""
    if not path.exists():
        msg = f"index manifest {path} is missing; run `medleaf index build`"
        raise IndexMissingError(msg)
    try:
        return IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as error:
        msg = f"index manifest {path} is unreadable: {error}"
        raise IndexMissingError(msg) from error


def build_manifest(
    *,
    collection: str,
    corpus_version: str,
    corpus_sha256_value: str,
    encoder: Encoder,
    dense_vector_name: str,
    sparse_vector_name: str,
    chunk_counts: dict[str, int],
    medicines: Sequence[str],
    languages: Sequence[str],
    missing: Sequence[str],
    active_substances: Mapping[str, str | None] | None = None,
    atc_codes: Mapping[str, str | None] | None = None,
    missing_metadata_entries: Sequence[str] = (),
    split_sections: Mapping[str, int] | None = None,
) -> IndexManifest:
    """Assemble the manifest for a finished index build."""
    return IndexManifest(
        collection=collection,
        corpus_version=corpus_version,
        corpus_sha256=corpus_sha256_value,
        embed_model=encoder.model_id,
        embed_revision=encoder.revision,
        dense_dim=encoder.dimension,
        dense_vector_name=dense_vector_name,
        sparse_vector_name=sparse_vector_name,
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        total_chunks=sum(chunk_counts.values()),
        chunk_counts=chunk_counts,
        medicines=list(medicines),
        languages=list(languages),
        missing=list(missing),
        active_substances=dict(active_substances or {}),
        atc_codes=dict(atc_codes or {}),
        missing_metadata=list(missing_metadata_entries),
        split_sections=dict(split_sections or {}),
    )


__all__ = [
    "DEFAULT_PREFETCH_LIMIT",
    "MANIFEST_SCHEMA_VERSION",
    "POINT_NAMESPACE",
    "IndexIncompleteError",
    "IndexManifest",
    "IndexMissingError",
    "RetrievalUnavailableError",
    "alias_target",
    "build_collection_name",
    "build_filter",
    "build_manifest",
    "check_complete",
    "collection_count",
    "corpus_sha256",
    "corpus_version_for",
    "delete_medicine_points",
    "ensure_collection",
    "hybrid_search",
    "load_manifest",
    "medicine_metadata",
    "missing_metadata",
    "payload_for",
    "point_alias",
    "point_id",
    "require_index",
    "set_medicine_metadata",
    "upsert_sections",
    "write_manifest",
]
