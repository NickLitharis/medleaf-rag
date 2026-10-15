"""Chunking: one chunk per numbered section, with tables flattened to readable rows.

SPEC section 8:

* One chunk = one section, e.g. SmPC 4.3.
* Sections over ~400 tokens are split at subsection or paragraph boundaries.
* Side-effect tables become one line per row, e.g. ``Nausea - very common (>=1/10)``,
  and stay in the same chunk as their heading.
* Chunk IDs are stable and language-independent: ``eliquis:smpc:4.5``.

A chunk is never allowed to silently outgrow the embedding window. The index build
passes the real e5 tokenizer as :class:`ChunkBudget.count_tokens`, so a long section is
split at table-row, paragraph, sentence or word boundaries until every part fits the
model window; the text is partitioned, never truncated.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import fitz  # PyMuPDF

from medleaf.models import DocKind, Lang, Section

#: A cell is "bold" when the first non-empty text span inside it has PyMuPDF's bold
#: flag (bit 4 of ``span['flags']``). EMA system-organ-class headings are bold; the
#: reaction names under them are not, and that is the only thing that separates a
#: heading row from a reaction that sits only in the first frequency column.
_BOLD_FLAG = 1 << 4

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


def word_count(text: str) -> int:
    """Word-count budget, the original M1 behaviour and the default for non-embedding callers."""
    return len(text.split())


def approximate_token_count(text: str) -> int:
    """Deliberately conservative token estimate used only when no tokenizer is available.

    Assumes at most two characters per token, which over-counts Latin text and matches
    Greek closely, so it can only over-split — never let a chunk exceed the window. The
    index build does not use this: it passes the real e5 tokenizer.
    """
    return max(len(text.split()), (len(text) + 1) // 2)


@dataclass(frozen=True)
class ChunkBudget:
    """How many tokens a chunk may contain, and how that is measured.

    ``reserve_tokens`` leaves room for the ``passage: `` prefix and the model's special
    tokens. ``label`` is part of the parse-cache key: a cache built with a different
    counter or window must be re-parsed rather than reused.
    """

    max_tokens: int = 512
    reserve_tokens: int = 16
    count_tokens: Callable[[str], int] = approximate_token_count
    label: str = "approximate"

    @property
    def limit(self) -> int:
        """Token budget one chunk part must stay within."""
        return max(1, self.max_tokens - self.reserve_tokens)

    def fits(self, text: str) -> bool:
        """True when ``text`` fits the budget."""
        return self.count_tokens(text) <= self.limit


#: Sentence boundary: keeps the terminator with the sentence it closes.
_SENTENCE_RE = re.compile(r"(?<=[.;:!?])\s+")


def _word_windows(sentence: str, budget: ChunkBudget) -> list[str]:
    """Last resort: split a single oversized sentence into whole-word windows.

    A word longer than the whole budget is emitted on its own; splitting inside a word
    would corrupt the clinical text.
    """
    windows: list[str] = []
    buffer: list[str] = []
    for word in sentence.split():
        candidate = " ".join([*buffer, word])
        if buffer and not budget.fits(candidate):
            windows.append(" ".join(buffer))
            buffer = [word]
        else:
            buffer.append(word)
    if buffer:
        windows.append(" ".join(buffer))
    return windows


def split_units(text: str, budget: ChunkBudget) -> list[str]:
    """Split section text into the smallest units a chunk may be cut at.

    Table rows (one line each in the flattened output) stay intact, then paragraphs,
    then sentences, then whole words. Nothing is dropped: joining the units with a
    newline and collapsing whitespace reproduces the original text.
    """
    units: list[str] = []
    for block in text.split("\n\n"):
        for raw_line in block.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            if budget.fits(line):
                units.append(line)
                continue
            for raw_sentence in _SENTENCE_RE.split(line):
                sentence = raw_sentence.strip()
                if not sentence:
                    continue
                if budget.fits(sentence):
                    units.append(sentence)
                else:
                    units.extend(_word_windows(sentence, budget))
    return units


def split_long_text(text: str, budget: ChunkBudget) -> list[str]:
    """Pack a section into parts that each fit the budget.

    Unlike the original word-count version, an oversized paragraph is no longer emitted
    whole: it is split at sentence and word boundaries so the embedding model never has
    to truncate clinical content silently.
    """
    units = split_units(text, budget)
    if not units:
        return []
    parts: list[str] = []
    buffer: list[str] = []
    for unit in units:
        if buffer and not budget.fits("\n".join([*buffer, unit])):
            parts.append("\n".join(buffer))
            buffer = [unit]
        else:
            buffer = [*buffer, unit]
    if buffer:
        parts.append("\n".join(buffer))
    return parts


def _is_frequency_label(cell: str) -> bool:
    """True when a header cell names a frequency band (e.g. 'Common', 'Σπάνιες')."""
    return bool(FREQUENCY_RE.search(cell))


def flatten_table_rows(
    rows: list[list[str | None]],
    bolds: list[list[bool | None]] | None = None,
) -> list[str]:
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

    A grid row with a single value in the first column is either a system-organ-class
    heading or a reaction that only ever appears in the first frequency column. Font
    weight is the only reliable discriminator: ``bolds`` carries the per-cell bold flag
    from the page text (``True`` bold, ``False`` regular, ``None`` unknown). A bold
    first cell is kept as a heading; a regular one is emitted as a reaction with the
    first column's frequency. When ``bolds`` is absent (or the flag is unknown) the row
    is treated as a heading, which is safe but loses the frequency — callers that have
    the page should pass ``bolds``.
    """
    if not rows:
        return []

    header = [_clean_cell(cell) for cell in rows[0]]
    grid_form = bool(header) and bool(header[0]) and _is_frequency_label(header[0])

    lines: list[str] = []
    current_reaction = ""
    for row_index, row in enumerate(rows[1:], start=1):
        cells = [_clean_cell(cell) for cell in row]
        if not any(cells):
            continue

        if grid_form:
            present = [(index, value) for index, value in enumerate(cells) if value]
            if len(present) == 1 and present[0][0] == 0:
                if _cell_is_regular(bolds, row_index, 0):
                    # A reaction that lives only in the first frequency column.
                    label = header[0] if header else ""
                    lines.append(f"{cells[0]} - {label}" if label else cells[0])
                else:
                    # System-organ-class heading (bold, or weight unknown).
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


def _cell_is_regular(
    bolds: list[list[bool | None]] | None,
    row_index: int,
    column_index: int,
) -> bool:
    """True only when the cell is known to be non-bold (a reaction, not a heading)."""
    if bolds is None or row_index >= len(bolds):
        return False
    row = bolds[row_index]
    if column_index >= len(row):
        return False
    return row[column_index] is False


def _page_bold_spans(page: fitz.Page) -> list[tuple[fitz.Rect, bool]]:
    """Text spans on a page as ``(bbox, is_bold)``, extracted once per page."""
    spans: list[tuple[fitz.Rect, bool]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", ()):
            for span in line["spans"]:
                if span["text"].strip():
                    spans.append((fitz.Rect(span["bbox"]), bool(span["flags"] & _BOLD_FLAG)))
    return spans


def _cell_bold_flags(
    table_rows: Any,
    spans: list[tuple[fitz.Rect, bool]],
) -> list[list[bool | None]]:
    """Bold flag per table cell, in the same row/cell order as ``table.extract()``.

    PyMuPDF's cell rectangles touch neighbouring text spans by a fraction of a point,
    so a plain ``intersects`` test picks the wrong span at a row boundary. The span with
    the largest overlap area is the cell's real text; a cell with no overlapping span is
    ``None`` (PyMuPDF reports merged/empty cells with no geometry), and an unknown weight
    must not be read as a reaction.
    """
    flags: list[list[bool | None]] = []
    for row in table_rows:
        row_flags: list[bool | None] = []
        for cell in row.cells:
            if cell is None:
                row_flags.append(None)
                continue
            rect = fitz.Rect(cell)
            bold: bool | None = None
            best_area = 0.0
            for span_rect, span_bold in spans:
                area = (rect & span_rect).get_area()
                if area > best_area:
                    best_area = area
                    bold = span_bold
            row_flags.append(bold)
        flags.append(row_flags)
    return flags


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
            spans = _page_bold_spans(page)
            lines: list[str] = []
            for table in page.find_tables().tables:
                rows = table.extract()
                table_rows = getattr(table, "rows", None)
                bolds = _cell_bold_flags(table_rows, spans) if table_rows is not None else None
                lines.extend(flatten_table_rows(rows, bolds))
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
    max_tokens: int = 400,
    budget: ChunkBudget | None = None,
) -> list[Section]:
    """Convert parser output into chunked :class:`Section` objects.

    Sections longer than the budget produce numbered part IDs in the ``chunk_id``
    (``4.5#2``) so IDs stay stable and unique. ``Section.section`` stays the base
    number (``4.5``) so section coverage is measured per document section, not per part.

    ``budget`` wins over ``max_tokens``; ``max_tokens`` is the word-count fallback used
    by callers that do not embed (and by the M1 tests).
    """
    active = budget or ChunkBudget(max_tokens=max_tokens, count_tokens=word_count, label="words")
    sections: list[Section] = []
    for number, title, text in raw_sections:
        body = text.strip()
        parts = split_long_text(body, active) if body else [""]
        total = len(parts)
        for index, part in enumerate(parts, start=1):
            sections.append(
                Section(
                    chunk_id=(
                        chunk_id(medicine, doc, number)
                        if total == 1
                        else f"{chunk_id(medicine, doc, number)}#{index}"
                    ),
                    medicine=medicine,
                    lang=lang,
                    doc=doc,
                    section=number,
                    title=title if total == 1 else f"{title} (part {index})",
                    text=part,
                    source_url=source_url,
                    revision_date=revision_date,
                    corpus_version=corpus_version,
                    part=index,
                    parts_total=total,
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
