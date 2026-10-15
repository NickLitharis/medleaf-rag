"""Chunking: one chunk per numbered section, with tables flattened to readable rows.

SPEC section 8:

* One chunk = one section, e.g. SmPC 4.3.
* Sections over ~400 tokens are split at subsection or paragraph boundaries.
* Side-effect tables become one line per row, e.g. ``Nausea - very common (>=1/10)``,
  and stay in the same chunk as their heading.
* Chunk IDs are stable and language-independent: ``eliquis:smpc:4.5``.
"""

from __future__ import annotations

import re

import fitz  # PyMuPDF

from medleaf.models import DocKind, Lang, Section

# Table detection: a paragraph made of pipe-separated cells produced by the flattener.
FREQUENCY_WORDS = {
    "very common",
    "common",
    "uncommon",
    "rare",
    "very rare",
    "not known",
    "πολύ συχνές",
    "συχνές",
    "όχι συχνές",
    "σπάνιες",
    "πολύ σπάνιες",
    "μη γνωστές",
}
FREQUENCY_RE = re.compile(
    r"(very common|common|uncommon|rare|very rare|not known"
    r"|πολ[υύ] συχν[εέ]ς|συχν[εέ]ς|[οό]χι συχν[εέ]ς"
    r"|σπ[αά]νι[εέ]ς|πολ[υύ] σπ[αά]νι[εέ]ς|μη γνωστ[εέ]ς)",
    re.IGNORECASE,
)
FREQUENCY_QUALIFIER_RE = re.compile(r"\(\s*(≥|>|<=|<|> =)?\s*1\s*/\s*([0-9]{1,3})\s*\)")


def chunk_id(medicine: str, doc: DocKind, section: str) -> str:
    """Stable, language-independent chunk identifier (SPEC section 8)."""
    return f"{medicine}:{doc.value}:{section}"


def split_long_text(text: str, max_tokens: int) -> list[str]:
    """Split a long section at paragraph boundaries, keeping paragraphs intact.

    A single paragraph longer than ``max_tokens`` is emitted whole rather than cut
    mid-sentence: a truncated clinical statement is worse than a long chunk.
    """
    paragraphs = [block.strip() for block in text.split("\n\n") if block.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    count = 0
    for paragraph in paragraphs:
        length = len(paragraph.split())
        if buffer and count + length > max_tokens:
            chunks.append("\n\n".join(buffer))
            buffer, count = [], 0
        buffer.append(paragraph)
        count += length
    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def _is_frequency_label(cell: str) -> bool:
    """True when a header cell names a frequency band (e.g. 'Common', 'Σπάνιες')."""
    return bool(FREQUENCY_RE.search(cell))


def flatten_table_rows(rows: list[list[str]]) -> list[str]:
    """Turn a side-effect table into one readable line per reaction/frequency pair.

    EMA ships two table shapes, and they need different flattening:

    * *labelled* — the first header cell is a reaction/SOC label and the remaining
      headers are frequency bands (``Reaction | Frequency``). A row becomes
      ``Reaction - frequency``.
    * *grid* — the first header cell is itself a frequency band
      (``Common | Uncommon | Rare | ...``). The column position carries the frequency,
      so every non-empty cell is a reaction and becomes ``Reaction - <column header>``.
      Treating the first cell as a reaction here is what previously produced lies like
      ``Anaemia - Thrombocytosis`` (two different frequencies glued together).

    A row with a single value in the first column of a grid is a system-organ-class
    heading, not a reaction.
    """
    if not rows:
        return []

    header = [_clean_cell(cell) for cell in rows[0]]
    grid_form = bool(header) and bool(header[0]) and _is_frequency_label(header[0])

    lines: list[str] = []
    current_reaction = ""
    for row in rows[1:]:
        cells = [_clean_cell(cell) for cell in row]
        if not any(cells):
            continue

        if grid_form:
            present = [(index, value) for index, value in enumerate(cells) if value]
            if len(present) == 1 and present[0][0] == 0:
                # System-organ-class heading row (or a reaction with no other band).
                lines.append(cells[0])
                continue
            for index, value in present:
                label = header[index] if index < len(header) else ""
                lines.append(f"{value} - {label}" if label else value)
            continue

        reaction = cells[0]
        if reaction:
            current_reaction = reaction
        elif current_reaction and any(cells[1:]):
            # Continuation row: the label lives on the row above (merged cell).
            reaction = current_reaction
        else:
            continue

        values = [(header[i] if i < len(header) else "", cells[i]) for i in range(1, len(cells))]
        values = [(name, value) for name, value in values if value]
        if not values:
            # A system-organ-class heading row: keep it as a heading line.
            lines.append(reaction)
            continue
        if len(values) == 1:
            lines.append(f"{reaction} - {values[0][1]}")
        else:
            joined = "; ".join(f"{name}: {value}" for name, value in values)
            lines.append(f"{reaction} - {joined}")
    return lines


def _clean_cell(cell: str | None) -> str:
    if not cell:
        return ""
    return re.sub(r"\s+", " ", cell.replace("\n", " ")).strip()


def extract_tables_for_pages(pdf_path: str, pages: set[int]) -> dict[int, list[str]]:
    """Extract flattened table lines for the given 1-based page numbers."""
    result: dict[int, list[str]] = {}
    with fitz.open(pdf_path) as document:
        for page_number in sorted(pages):
            if page_number < 1 or page_number > document.page_count:
                continue
            page = document[page_number - 1]
            lines: list[str] = []
            for table in page.find_tables().tables:
                lines.extend(flatten_table_rows(table.extract()))
            result[page_number] = lines
    return result


def append_table_lines(text: str, table_lines: list[str]) -> str:
    """Append flattened table rows to section text, skipping ones already present."""
    if not table_lines:
        return text
    existing = text
    additions = [line for line in table_lines if line and line not in existing]
    if not additions:
        return text
    return text + "\n\n" + "\n".join(additions)


def build_sections(
    *,
    medicine: str,
    lang: Lang,
    doc: DocKind,
    raw_sections: list[tuple[str, str, str]],
    source_url: str,
    revision_date: str | None,
    corpus_version: str,
    max_tokens: int,
) -> list[Section]:
    """Convert parser output into chunked :class:`Section` objects.

    Sections longer than ``max_tokens`` produce numbered part IDs in the ``chunk_id``
    (``4.5#2``) so IDs stay stable and unique. ``Section.section`` stays the base
    number (``4.5``) so section coverage is measured per document section, not per part.
    """
    sections: list[Section] = []
    for number, title, text in raw_sections:
        body = text.strip()
        if not body:
            # Keep the section so coverage and citation lookups still resolve; an empty
            # body is honest about a heading with no text under it.
            body = ""
        parts = split_long_text(body, max_tokens) if body else [""]
        if len(parts) == 1:
            sections.append(
                Section(
                    chunk_id=chunk_id(medicine, doc, number),
                    medicine=medicine,
                    lang=lang,
                    doc=doc,
                    section=number,
                    title=title,
                    text=body,
                    source_url=source_url,
                    revision_date=revision_date,
                    corpus_version=corpus_version,
                )
            )
            continue
        for index, part in enumerate(parts, start=1):
            sections.append(
                Section(
                    chunk_id=f"{chunk_id(medicine, doc, number)}#{index}",
                    medicine=medicine,
                    lang=lang,
                    doc=doc,
                    section=number,
                    title=f"{title} (part {index})",
                    text=part,
                    source_url=source_url,
                    revision_date=revision_date,
                    corpus_version=corpus_version,
                )
            )
    return sections


def is_frequency_line(line: str) -> bool:
    """True when a flattened table line carries an adverse-reaction frequency."""
    return bool(FREQUENCY_RE.search(line))


def is_readable_frequency_row(line: str, *, max_chars: int = 1000) -> bool:
    """True for a flattened row that reads as ``Reaction - frequency``.

    A run-on section paragraph can contain a frequency word by accident, so a genuine
    row must also contain the ``' - '`` separator the flattener emits and must be far
    shorter than a reflowed block (real multi-indication rows run to a few hundred
    characters; a joined section paragraph is several thousand).
    """
    return is_frequency_line(line) and " - " in line and len(line) <= max_chars
