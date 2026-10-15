"""Parse cache and Qdrant index build (``medleaf index build``).

The parse is the slow, PDF-layout-sensitive step, so its output is cached per PDF
checksum in ``data/parsed``. A cached parse is reused only when the PDF bytes, the
parser/cache version and the chunking budget label all match; a parsing fix or a
different embedding window therefore re-parses instead of silently reusing stale
chunks.

Chunks are always stamped with the *current* corpus version, whether they came from the
cache or from a fresh parse, so a version switch cannot leave provenance that disagrees
with the version filter the search applies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient

from medleaf.config import Settings
from medleaf.ingest.chunk import ChunkBudget
from medleaf.ingest.download import sha256_of
from medleaf.ingest.pipeline import PARSER_CACHE_VERSION, parse_pdf
from medleaf.ingest.store import load_medicines
from medleaf.models import Lang, Medicine, Section
from medleaf.retrieval.embed import Encoder
from medleaf.retrieval.index import (
    IndexManifest,
    build_manifest,
    corpus_sha256,
    corpus_version_for,
    delete_medicine_points,
    ensure_collection,
    load_manifest,
    medicine_metadata,
    missing_metadata,
    set_medicine_metadata,
    upsert_sections,
    write_manifest,
)

#: Tokens reserved for the ``passage: `` prefix and the model's special tokens.
BUDGET_RESERVE_TOKENS = 16


class IndexBuildError(RuntimeError):
    """The index cannot be built as requested (usually missing source documents)."""


class ParsedCache(BaseModel):
    """Cached parse of one PDF, valid only for the inputs it was parsed from."""

    model_config = ConfigDict(extra="forbid")

    cache_version: str
    pdf_sha256: str
    budget_label: str
    corpus_version: str
    source_url: str
    sections: list[Section]


def cache_is_valid(
    cached: ParsedCache | None,
    *,
    pdf_sha256: str,
    budget_label: str,
) -> bool:
    """True only when a cached parse matches the current PDF, parser and window.

    The parser/cache version is what makes a parsing or chunking fix invalidate old
    caches; the budget label does the same for a change of embedding model or window.
    """
    if cached is None:
        return False
    return (
        cached.cache_version == PARSER_CACHE_VERSION
        and cached.pdf_sha256 == pdf_sha256
        and cached.budget_label == budget_label
    )


def resolved_corpus_version(settings: Settings) -> str:
    """The corpus version to stamp on chunks: the explicit one, else the manifest hash."""
    if settings.corpus_version != "dev":
        return settings.corpus_version
    return corpus_version_for(settings.medicines_file)


def chunk_budget_for(encoder: Encoder, settings: Settings) -> ChunkBudget:
    """Build the chunk budget from the real tokenizer and the model window."""
    cache: dict[str, int] = {}

    def count(text: str) -> int:
        value = cache.get(text)
        if value is None:
            value = encoder.count_tokens([text])[0]
            cache[text] = value
        return value

    return ChunkBudget(
        max_tokens=settings.embed_max_seq_length,
        reserve_tokens=BUDGET_RESERVE_TOKENS,
        count_tokens=count,
        label=f"tokenizer:{encoder.model_id}@{encoder.revision}",
    )


@dataclass(frozen=True)
class ParseResult:
    """One parsed PDF and where the section list came from."""

    medicine: Medicine
    lang: Lang
    sections: list[Section]
    origin: str  # "cache" or "parsed"

    @property
    def key(self) -> str:
        return f"{self.medicine.slug}:{self.lang.value}"


def _stamp(sections: Sequence[Section], corpus_version: str) -> list[Section]:
    """Return the sections with the current corpus version stamped on every chunk."""
    return [
        section
        if section.corpus_version == corpus_version
        else section.model_copy(update={"corpus_version": corpus_version})
        for section in sections
    ]


def _read_cache(path: Path) -> ParsedCache | None:
    """Read a cache file, returning ``None`` when it is unreadable or outdated in shape."""
    try:
        return ParsedCache.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def parse_with_cache(
    settings: Settings,
    medicine: Medicine,
    lang: Lang,
    *,
    budget: ChunkBudget,
    reuse_cache: bool = True,
) -> ParseResult:
    """Return the chunks for one PDF, parsing only when the cache is not valid."""
    pdf_path = settings.pdf_path(medicine.slug, lang.value)
    if not pdf_path.exists():
        msg = f"missing source PDF {pdf_path}"
        raise IndexBuildError(msg)
    corpus_version = resolved_corpus_version(settings)
    digest = sha256_of(pdf_path)
    cache_path = settings.parsed_path(medicine.slug, lang.value)
    if reuse_cache and cache_path.exists():
        cached = _read_cache(cache_path)
        if cache_is_valid(cached, pdf_sha256=digest, budget_label=budget.label):
            assert cached is not None  # narrowed by cache_is_valid
            return ParseResult(
                medicine, lang, _stamp(cached.sections, corpus_version), origin="cache"
            )

    parsed = parse_pdf(
        medicine=medicine,
        lang=lang,
        pdf_path=pdf_path,
        corpus_version=corpus_version,
        max_tokens=settings.max_section_tokens,
        budget=budget,
    )
    cache = ParsedCache(
        cache_version=PARSER_CACHE_VERSION,
        pdf_sha256=digest,
        budget_label=budget.label,
        corpus_version=corpus_version,
        source_url=parsed.source_url,
        sections=parsed.sections,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(cache.model_dump_json(), encoding="utf-8")
    return ParseResult(
        medicine, lang, _stamp(parsed.sections, corpus_version), origin="parsed"
    )


def parse_corpus(
    settings: Settings,
    medicines: Sequence[Medicine],
    *,
    budget: ChunkBudget,
    reuse_cache: bool = True,
    progress: Callable[[str], None] = lambda _line: None,
) -> tuple[list[ParseResult], list[str]]:
    """Parse every requested medicine/language; return results and missing keys."""
    results: list[ParseResult] = []
    missing: list[str] = []
    for medicine in medicines:
        for lang in (Lang.EN, Lang.EL):
            try:
                result = parse_with_cache(
                    settings, medicine, lang, budget=budget, reuse_cache=reuse_cache
                )
            except IndexBuildError:
                missing.append(f"{medicine.slug}:{lang.value}")
                progress(f"MISSING {medicine.slug} {lang.value}: no source PDF")
                continue
            results.append(result)
            progress(
                f"{result.origin:6} {medicine.slug} {lang.value}: {len(result.sections)} chunks"
            )
    return results, missing


def split_section_counts(results: Sequence[ParseResult]) -> dict[str, int]:
    """Sections that were split into parts, keyed ``slug:lang:doc:section``."""
    counts: dict[str, int] = {}
    for result in results:
        for section in result.sections:
            if section.parts_total > 1:
                key = (
                    f"{section.medicine}:{section.lang.value}:"
                    f"{section.doc.value}:{section.section}"
                )
                counts[key] = max(counts.get(key, 1), section.parts_total)
    return counts


@dataclass(frozen=True)
class WindowReport:
    """How the indexed chunks sit against the embedding window."""

    collection: str
    total_points: int
    over_window: int
    limit_tokens: int
    worst: list[tuple[str, int]]
    by_medicine: dict[str, int]

    @property
    def clean(self) -> bool:
        """True when no indexed chunk exceeds the model window."""
        return self.over_window == 0


def window_report(
    *,
    settings: Settings,
    client: QdrantClient,
    encoder: Encoder,
    worst_limit: int = 10,
    scroll_page: int = 256,
) -> WindowReport:
    """Count the real tokens of every indexed chunk against the model window.

    This is the measurement behind the claim that clinical text is not truncated: an
    index is only complete when no chunk exceeds the window the model can see. It reads
    payloads only, so it can be run before and after a targeted rebuild.
    """
    limit_tokens = settings.embed_max_seq_length - BUDGET_RESERVE_TOKENS
    offset: object | None = None
    total = 0
    over = 0
    worst: list[tuple[str, int]] = []
    by_medicine: dict[str, int] = {}
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=scroll_page,
            offset=offset,
            with_payload=["chunk_id", "medicine", "text"],
            with_vectors=False,
        )
        texts = [str((point.payload or {}).get("text", "")) for point in points]
        counts = encoder.count_tokens(texts) if texts else []
        for point, count in zip(points, counts, strict=True):
            payload = point.payload or {}
            total += 1
            worst.append((str(payload.get("chunk_id", point.id)), count))
            if count > limit_tokens:
                over += 1
                slug = str(payload.get("medicine", "?"))
                by_medicine[slug] = by_medicine.get(slug, 0) + 1
        if offset is None:
            break
    worst.sort(key=lambda item: item[1], reverse=True)
    return WindowReport(
        collection=settings.qdrant_collection,
        total_points=total,
        over_window=over,
        limit_tokens=limit_tokens,
        worst=worst[:worst_limit],
        by_medicine=by_medicine,
    )


def merge_manifest(
    previous: IndexManifest | None,
    *,
    rebuilt_slugs: Sequence[str],
    current: IndexManifest,
) -> IndexManifest:
    """Fold a targeted rebuild into the previous manifest.

    A targeted build only knows the medicines it rebuilt, so its chunk counts and split
    sections are merged into the previous manifest instead of replacing the whole
    corpus record with a partial one.
    """
    if previous is None:
        return current
    prefixes = tuple(f"{slug}:" for slug in rebuilt_slugs)

    def keep(key: str) -> bool:
        return not key.startswith(prefixes)

    chunk_counts = {
        key: value for key, value in previous.chunk_counts.items() if keep(key)
    }
    split_sections = {
        key: value for key, value in previous.split_sections.items() if keep(key)
    }
    chunk_counts.update(current.chunk_counts)
    split_sections.update(current.split_sections)
    return current.model_copy(
        update={
            "chunk_counts": chunk_counts,
            "total_chunks": sum(chunk_counts.values()),
            "split_sections": split_sections,
            "medicines": sorted(set(previous.medicines) | set(current.medicines)),
            "missing": [entry for entry in previous.missing if keep(entry)],
            "active_substances": {**previous.active_substances, **current.active_substances},
            "atc_codes": {**previous.atc_codes, **current.atc_codes},
            "missing_metadata": sorted(
                {entry for entry in previous.missing_metadata if keep(entry)}
                | set(current.missing_metadata)
            ),
        }
    )


def build_index(
    *,
    settings: Settings,
    client: QdrantClient,
    encoder: Encoder,
    only: Sequence[str] | None = None,
    recreate: bool = True,
    reuse_cache: bool = True,
    allow_missing: bool = False,
    progress: Callable[[str], None] = print,
) -> IndexManifest:
    """Parse the corpus, embed every chunk and upsert it into Qdrant.

    Missing source documents abort the build unless ``allow_missing`` is set; when it
    is, they are recorded in the manifest's ``missing`` list and never counted as
    indexed chunks.

    A targeted build (``only`` without ``recreate``) deletes those medicines' existing
    points first, so chunk IDs from an older parse or an older chunking cannot linger
    next to the fresh ones. Embedding is per medicine, so a targeted rebuild re-embeds
    only the medicines it was asked for.
    """
    all_medicines = load_medicines(settings.medicines_file)
    if not all_medicines:
        msg = f"no medicines in {settings.medicines_file}; run `medleaf resolve` first"
        raise IndexBuildError(msg)
    requested = set(only or [])
    selected = [m for m in all_medicines if m.slug in requested] if only else list(all_medicines)
    if not selected:
        msg = f"none of {sorted(requested)} are in {settings.medicines_file}"
        raise IndexBuildError(msg)

    budget = chunk_budget_for(encoder, settings)
    progress(f"chunk budget: {budget.label}, {budget.max_tokens} tokens ({budget.limit} usable)")
    results, missing = parse_corpus(
        settings, selected, budget=budget, reuse_cache=reuse_cache, progress=progress
    )
    if missing and not allow_missing:
        msg = (
            "missing source documents for "
            + ", ".join(missing)
            + "; download them (`medleaf download`) or pass allow_missing=True"
        )
        raise IndexBuildError(msg)

    corpus_version = resolved_corpus_version(settings)
    dimension = encoder.dimension
    created = ensure_collection(
        client,
        collection=settings.qdrant_collection,
        dense_dim=dimension,
        dense_vector_name=settings.dense_vector_name,
        sparse_vector_name=settings.sparse_vector_name,
        recreate=recreate,
    )
    progress(
        f"collection {settings.qdrant_collection!r} "
        f"{'created' if created else 'reused'} (dense dim {dimension}, sparse IDF)"
    )

    if only and not recreate:
        for medicine in selected:
            removed = delete_medicine_points(
                client, collection=settings.qdrant_collection, slug=medicine.slug
            )
            progress(f"replaced {medicine.slug}: removed {removed} stale point(s)")

    metadata = {medicine.slug: medicine_metadata(medicine) for medicine in selected}
    chunk_counts: dict[str, int] = {}
    batch_size = settings.embed_batch_size
    for result in results:
        for start in range(0, len(result.sections), batch_size):
            batch = result.sections[start : start + batch_size]
            vectors = encoder.encode_passages([section.text for section in batch])
            upsert_sections(
                client,
                collection=settings.qdrant_collection,
                sections=batch,
                dense_vectors=vectors,
                dense_vector_name=settings.dense_vector_name,
                sparse_vector_name=settings.sparse_vector_name,
                metadata=metadata,
                wait=start + batch_size >= len(result.sections),
            )
        chunk_counts[result.key] = len(result.sections)
        progress(f"indexed {result.key}: {len(result.sections)} chunks")

    split_sections = split_section_counts(results)
    manifest = build_manifest(
        collection=settings.qdrant_collection,
        corpus_version=corpus_version,
        corpus_sha256_value=corpus_sha256(settings.medicines_file),
        encoder=encoder,
        dense_vector_name=settings.dense_vector_name,
        sparse_vector_name=settings.sparse_vector_name,
        chunk_counts=chunk_counts,
        medicines=[m.slug for m in selected],
        languages=[lang.value for lang in (Lang.EN, Lang.EL)],
        missing=missing,
        active_substances={m.slug: m.active_substance or None for m in selected},
        atc_codes={m.slug: m.atc_code or None for m in selected},
        missing_metadata_entries=missing_metadata(selected),
        split_sections=split_sections,
    )
    if only and not recreate and settings.index_manifest_file.exists():
        manifest = merge_manifest(
            load_manifest(settings.index_manifest_file),
            rebuilt_slugs=[m.slug for m in selected],
            current=manifest,
        )
    write_manifest(settings.index_manifest_file, manifest)
    progress(
        f"manifest {settings.index_manifest_file}: {manifest.total_chunks} chunks, "
        f"corpus_version={manifest.corpus_version}, model={manifest.embed_model}@"
        f"{manifest.embed_revision[:12]}, split sections={len(split_sections)}, "
        f"missing metadata={len(manifest.missing_metadata)}"
    )
    return manifest


def annotate_manifest_metadata(
    *,
    settings: Settings,
    client: QdrantClient,
    progress: Callable[[str], None] = print,
) -> Mapping[str, int]:
    """Attach manifest metadata to an existing index without re-embedding.

    Used when the payload needs the SPEC section 8 metadata but the vectors are already
    correct: the points are updated in place and the manifest records the coverage.
    """
    medicines = load_medicines(settings.medicines_file)
    if not medicines:
        msg = f"no medicines in {settings.medicines_file}; run `medleaf resolve` first"
        raise IndexBuildError(msg)
    manifest = load_manifest(settings.index_manifest_file)
    metadata = {medicine.slug: medicine_metadata(medicine) for medicine in medicines}
    updated = set_medicine_metadata(
        client, collection=settings.qdrant_collection, metadata_by_slug=metadata
    )
    refreshed = manifest.model_copy(
        update={
            "active_substances": {m.slug: m.active_substance or None for m in medicines},
            "atc_codes": {m.slug: m.atc_code or None for m in medicines},
            "missing_metadata": missing_metadata(medicines),
        }
    )
    write_manifest(settings.index_manifest_file, refreshed)
    progress(f"updated manifest metadata; points touched: {sum(updated.values())}")
    return updated


__all__ = [
    "BUDGET_RESERVE_TOKENS",
    "IndexBuildError",
    "ParseResult",
    "ParsedCache",
    "WindowReport",
    "annotate_manifest_metadata",
    "build_index",
    "cache_is_valid",
    "chunk_budget_for",
    "merge_manifest",
    "parse_corpus",
    "parse_with_cache",
    "resolved_corpus_version",
    "split_section_counts",
    "window_report",
]
