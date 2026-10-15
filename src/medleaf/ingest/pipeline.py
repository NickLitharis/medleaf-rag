"""The parse pipeline: PDF -> sections -> chunks, for one medicine and language."""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from medleaf.ingest.chunk import append_table_lines, build_sections, extract_tables_for_pages
from medleaf.ingest.pdf import (
    SMPC_HEADING_RE,
    RawLine,
    clean_line,
    extract_lines,
    find_annex_boundaries,
    normalize_heading_line,
    rebuild_paragraphs,
    split_sections,
)
from medleaf.models import DocKind, Lang, Medicine, ParsedDocument, Section
from medleaf.sections import SMPC_NUMBERS

#: SmPC sections whose pages can carry flattened side-effect tables (SPEC section 8).
#: Table detection is the expensive part of parsing, so it runs only on these spans.
TABLE_SECTIONS: tuple[str, ...] = ("4.8",)

#: A dotted table-of-contents leader, e.g. ``4.1 Therapeutic indications .... 5``.
_DOTTED_LEADER_RE = re.compile(r"\.{4,}\s*\d*$")


def pdf_revision_date(pdf_path: str) -> str | None:
    """Read the PDF's own modification date as a per-language revision marker.

    The two language versions of one medicine are often revised on different dates
    (SPEC section 17), so the date is stored per language rather than per medicine.
    """
    with fitz.open(pdf_path) as document:
        metadata = document.metadata or {}
    raw = metadata.get("modDate") or metadata.get("creationDate") or ""
    return _parse_pdf_date(str(raw))


def _parse_pdf_date(raw: str) -> str | None:
    """Convert a PDF date string like ``D:20260408021909-04'00'`` to an ISO date."""
    if not raw.startswith("D:") or len(raw) < 10:
        return None
    digits = raw[2:10]
    if not digits.isdigit():
        return None
    return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"


def parse_pdf(
    *,
    medicine: Medicine,
    lang: Lang,
    pdf_path: Path,
    corpus_version: str,
    max_tokens: int,
) -> ParsedDocument:
    """Parse one product-information PDF into chunked sections.

    Side-effect tables are flattened and appended to the section that contains them, so
    the frequency information stays next to its heading (SPEC section 8).
    """
    document_ref = medicine.documents[lang]
    source_url = str(document_ref.url)
    revision_date = pdf_revision_date(str(pdf_path)) or document_ref.revision_date

    lines = extract_lines(str(pdf_path))

    sections: list[Section] = []
    for doc_kind in (DocKind.SMPC, DocKind.LEAFLET):
        raw = split_sections(lines, doc=doc_kind)
        # Reflow body text into paragraphs *before* table rows are appended. Appending
        # first would let the reflow join every flattened row into one giant line,
        # losing the row boundaries that make side-effect tables readable.
        raw = [(number, title, rebuild_paragraphs(text)) for number, title, text in raw]
        if doc_kind is DocKind.SMPC:
            raw = _attach_tables(str(pdf_path), raw, lines)
        sections.extend(
            build_sections(
                medicine=medicine.slug,
                lang=lang,
                doc=doc_kind,
                raw_sections=raw,
                source_url=source_url,
                revision_date=revision_date,
                corpus_version=corpus_version,
                max_tokens=max_tokens,
            )
        )

    return ParsedDocument(
        medicine=medicine.slug,
        lang=lang,
        source_path=pdf_path,
        source_url=source_url,
        sections=sections,
    )


def _smpc_annex_end(lines: list[RawLine]) -> int:
    """Index where the SmPC (Annex I) ends: the first later annex marker, or EOF."""
    boundaries = find_annex_boundaries(lines)
    ends = [
        boundaries[key]
        for key in ("annex2", "annex3", "labelling", "leaflet")
        if key in boundaries
    ]
    return min(ends) if ends else len(lines)


def _section_spans(lines: list[RawLine], numbers: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    """Line-index spans for the given SmPC section numbers.

    A span runs from the section's heading line to the line before the next SmPC
    heading, and never past the end of Annex I. The span has a real end, so a page after
    the last wanted heading is not silently attributed to it.
    """
    wanted = set(numbers)
    if not wanted:
        return {}
    smpc_end = _smpc_annex_end(lines)

    headings: list[tuple[int, str]] = []
    seen: set[str] = set()
    for index in range(smpc_end):
        text = normalize_heading_line(clean_line(lines[index].text))
        if _DOTTED_LEADER_RE.search(text):
            continue
        match = SMPC_HEADING_RE.match(text)
        if match is None:
            continue
        number = match.group("num")
        if number not in SMPC_NUMBERS or number in seen:
            continue
        seen.add(number)
        headings.append((index, number))

    spans: dict[str, tuple[int, int]] = {}
    for position, (index, number) in enumerate(headings):
        if number not in wanted:
            continue
        end_index = headings[position + 1][0] - 1 if position + 1 < len(headings) else smpc_end - 1
        if end_index >= index:
            spans[number] = (index, end_index)
    return spans


def _table_owner_pages(
    lines: list[RawLine],
    raw_sections: list[tuple[str, str, str]],
) -> dict[int, str]:
    """Map each 1-based page that can hold a table to the section that owns it.

    Detection is limited to the spans of :data:`TABLE_SECTIONS`, so cover pages,
    annexes, packaging and leaflet pages are never table-scanned. A page a heading
    starts on belongs to that heading's section even when the previous section also has
    body text on it, so a table that begins beside its ``4.8`` heading stays with 4.8.
    """
    present = {number for number, _, _ in raw_sections}
    spans = _section_spans(lines, tuple(number for number in TABLE_SECTIONS if number in present))
    if not spans:
        return {}

    ordered = sorted(spans, key=lambda number: spans[number][0])
    owner_by_page: dict[int, str] = {}
    for number in ordered:
        start_index, end_index = spans[number]
        for page in range(lines[start_index].page, lines[end_index].page + 1):
            owner_by_page.setdefault(page, number)
    # A heading page belongs to the section that starts there, even if the previous
    # section also has body text on it.
    for number in ordered:
        owner_by_page[lines[spans[number][0]].page] = number
    return owner_by_page


def _attach_tables(
    pdf_path: str,
    raw_sections: list[tuple[str, str, str]],
    lines: list[RawLine],
) -> list[tuple[str, str, str]]:
    """Append flattened table rows to the SmPC section that owns their page.

    Only pages inside a :data:`TABLE_SECTIONS` span are handed to PyMuPDF's table
    detector, and the document is opened once for the whole pass.
    """
    if not raw_sections:
        return raw_sections

    owner_by_page = _table_owner_pages(lines, raw_sections)
    if not owner_by_page:
        return raw_sections

    tables_by_page = extract_tables_for_pages(pdf_path, set(owner_by_page))

    table_lines_for_section: dict[str, list[str]] = {number: [] for number, _, _ in raw_sections}
    for page_number, owner in owner_by_page.items():
        table_lines_for_section[owner].extend(tables_by_page.get(page_number, ()))

    updated: list[tuple[str, str, str]] = []
    for number, title, text in raw_sections:
        extra = table_lines_for_section.get(number, [])
        updated.append((number, title, append_table_lines(text, extra)))
    return updated
