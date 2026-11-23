"""Parse the corpus into chunks, with a cache keyed on the PDF bytes.

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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from medleaf.config import Settings
from medleaf.embed import Encoder
from medleaf.ingest.chunk import ChunkBudget
from medleaf.ingest.download import sha256_of
from medleaf.ingest.pipeline import PARSER_CACHE_VERSION, parse_pdf
from medleaf.ingest.store import corpus_version_for
from medleaf.models import Lang, Medicine, Section

#: Tokens reserved for the ``passage: `` prefix and the model's special tokens.
BUDGET_RESERVE_TOKENS = 16


class CorpusError(RuntimeError):
    """The index cannot be built as requested (usually missing source documents)."""


class ChecksumMismatchError(CorpusError):
    """A downloaded PDF does not match the SHA-256 pinned in ``medicines.yaml``."""


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
    allow_checksum_mismatch: bool = False,
) -> ParseResult:
    """Return the chunks for one PDF, parsing only when the cache is not valid.

    A pinned checksum is enforced before the parse cache is consulted, so a changed
    PDF cannot reach the index under the old corpus pin. ``allow_checksum_mismatch``
    is the explicit opt-out for an intentional, locally reviewed re-download.
    """
    pdf_path = settings.pdf_path(medicine.slug, lang.value)
    if not pdf_path.exists():
        msg = f"missing source PDF {pdf_path}"
        raise CorpusError(msg)
    corpus_version = resolved_corpus_version(settings)
    digest = sha256_of(pdf_path)
    document = medicine.documents.get(lang)
    pinned = document.sha256 if document is not None else None
    if pinned is not None and digest != pinned and not allow_checksum_mismatch:
        msg = (
            f"{pdf_path} has sha256 {digest}, but {medicine.slug}:{lang.value} is pinned "
            f"to {pinned}; re-download with `medleaf download`, update the pin, or pass "
            "--allow-checksum-mismatch after reviewing the change"
        )
        raise ChecksumMismatchError(msg)
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
    allow_checksum_mismatch: bool = False,
    progress: Callable[[str], None] = lambda _line: None,
) -> tuple[list[ParseResult], list[str]]:
    """Parse every requested medicine/language; return results and missing keys.

    A checksum mismatch is not a missing document: it aborts the whole build instead
    of being recorded in ``missing`` and silently skipped.
    """
    results: list[ParseResult] = []
    missing: list[str] = []
    for medicine in medicines:
        for lang in (Lang.EN, Lang.EL):
            try:
                result = parse_with_cache(
                    settings,
                    medicine,
                    lang,
                    budget=budget,
                    reuse_cache=reuse_cache,
                    allow_checksum_mismatch=allow_checksum_mismatch,
                )
            except ChecksumMismatchError:
                raise
            except CorpusError:
                missing.append(f"{medicine.slug}:{lang.value}")
                progress(f"MISSING {medicine.slug} {lang.value}: no source PDF")
                continue
            results.append(result)
            progress(
                f"{result.origin:6} {medicine.slug} {lang.value}: {len(result.sections)} chunks"
            )
    return results, missing
















__all__ = [
    "BUDGET_RESERVE_TOKENS",
    "ChecksumMismatchError",
    "CorpusError",
    "ParseResult",
    "ParsedCache",
    "cache_is_valid",
    "chunk_budget_for",
    "parse_corpus",
    "parse_with_cache",
    "resolved_corpus_version",
]
