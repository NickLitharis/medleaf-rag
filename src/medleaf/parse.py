"""PDF parsing into numbered sections (SPEC section 8).

One chunk is one numbered section. SmPC sections come from Annex I, leaflet
sections from Annex IIIB. Annex II and the labelling annex are skipped: they
carry packaging and administrative text, not answerable content.

Section numbering is identical in every EU language, so section identity is
language-independent and drives the shared chunk ID (SPEC section 8, step 5).

Headings are found from font weight, not from wording. A real numbered heading
is set in bold; a number that merely starts a line of body text ("10 mg twice
daily", a bullet "3. Complete...") is not. Wording heuristics cannot tell those
apart and, worse, differ per language, which is what a bilingual corpus must
not depend on.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import fitz

MAX_SECTION_TOKENS = 400
MIN_SECTION_CHARS = 80

#: Bold bit in the PyMuPDF span ``flags`` field.
_BOLD_FLAG = 1 << 4

#: A heading line is the section number alone, e.g. "4.1" or "4.1.".
_NUMBER_LINE_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)\.?$")
#: A heading line that also carries its title, e.g. "4.1 Therapeutic indications".
_NUMBER_TITLE_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)\.?\s+(\S.*)$")
#: A bare page number, e.g. "143".
_PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")
#: A dotted table-of-contents leader, e.g. "4.1 Therapeutic indications .... 5".
_DOTTED_LEADER_RE = re.compile(r"\.{4,}\s*\d*$")

_MAX_TITLE_CHARS = 160

#: Headings that end the SmPC body and must not be treated as sections.
_ANNEX_II_MARKERS = ("ANNEX II", "ΠΑΡΑΡΤΗΜΑ ΙΙ")
_ANNEX_III_MARKERS = ("ANNEX III", "ΠΑΡΑΡΤΗΜΑ III", "ΠΑΡΑΡΤΗΜΑ ΙΙΙ")
_LABELLING_MARKERS = ("LABELLING", "ΕΠΙΣΗΜΑΝΣΗ")
_LEAFLET_START_MARKERS = ("ANNEX III", "ΠΑΡΑΡΤΗΜΑ III", "ΠΑΡΑΡΤΗΜΑ ΙΙΙ")
_LEAFLET_MARKERS = ("PACKAGE LEAFLET", "ΦΥΛΛΟ ΟΔΗΓΙΩΝ ΧΡΗΣΗΣ")

SMPC_WANTED: tuple[str, ...] = (
    "4.1",
    "4.2",
    "4.3",
    "4.4",
    "4.5",
    "4.6",
    "4.7",
    "4.8",
    "4.9",
    "6.4",
)
LEAFLET_WANTED: tuple[str, ...] = ("1", "2", "3", "4", "5", "6")

#: Numbered headings that exist in each document kind. Section boundaries come
#: only from this vocabulary, so bold text that merely starts with a number
#: ("28 days to" in a table header, "10 mg twice daily") cannot cut a section
#: short. The vocabulary is numbering only, so it is language-independent.
SMPC_HEADINGS: tuple[str, ...] = (
    "1",
    "2",
    "3",
    "4",
    "7",
    "10",
    *(f"4.{i}" for i in range(1, 10)),
    *(f"5.{i}" for i in range(1, 4)),
    *(f"6.{i}" for i in range(1, 7)),
)
LEAFLET_HEADINGS: tuple[str, ...] = ("1", "2", "3", "4", "5", "6")


@dataclass(frozen=True, slots=True)
class Line:
    """One text line with the layout facts needed to classify it."""

    text: str
    page: int
    bold: bool
    size: float


@dataclass(frozen=True, slots=True)
class Section:
    """One numbered section of a product-information document."""

    doc: str
    section: str
    title: str
    text: str
    page: int

    @property
    def is_empty(self) -> bool:
        """True when the section holds no usable body text."""
        return len(self.text.strip()) < MIN_SECTION_CHARS


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Sections found in one PDF, plus a small diagnostics record."""

    doc: str
    lang: str
    sections: tuple[Section, ...]
    page_count: int
    page_table_counts: dict[int, int] = field(default_factory=dict)
    _pages: tuple[tuple[str, ...], ...] = ()

    def section(self, number: str) -> Section | None:
        """Return the section with this number, or None."""
        for candidate in self.sections:
            if candidate.section == number:
                return candidate
        return None

    def numbers(self) -> tuple[str, ...]:
        """Section numbers in document order."""
        return tuple(s.section for s in self.sections)

    def tables_on_page(self, page: int) -> int:
        """Number of tables detected on a 1-based page."""
        return self.page_table_counts.get(page, 0)

    def pages_with_tables(self) -> tuple[int, ...]:
        """1-based page numbers that contain at least one table."""
        return tuple(sorted(p for p, count in self.page_table_counts.items() if count > 0))

    def page_lines(self, page: int) -> tuple[str, ...]:
        """Raw text lines of a 1-based page (used by tests and diagnostics)."""
        if not self._pages or page < 1 or page > len(self._pages):
            return ()
        return self._pages[page - 1]


def _normalise(text: str) -> str:
    """Collapse layout whitespace while keeping paragraph breaks."""
    text = text.replace("\u00ad", "")
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _flatten_rows(rows: list[list[str]]) -> list[str]:
    """Flatten extracted table cells into one line per row (SPEC section 8, step 3).

    Row becomes ``label - value1; value2``: the first non-empty cell is the
    label and the rest are values, so a frequency stays attached to the
    reaction it belongs to (``Nausea - Common; Very common``).
    """
    flattened: list[str] = []
    for row in rows:
        cells = [_normalise(cell or "") for cell in row]
        filled = [c for c in cells if c]
        if len(filled) < 2:
            continue
        label, *values = filled
        if len(label) > _MAX_TITLE_CHARS:
            continue
        line = f"{label} - {'; '.join(values)}"
        if line.strip(" -;"):
            flattened.append(line)
    return flattened


def _tables_for_pages(
    pdf: fitz.Document,
    pages: list[int],
) -> tuple[dict[int, list[str]], dict[int, int]]:
    """Detect tables once per requested 1-based page and flatten their rows.

    Returns ``(rows_by_page, table_count_by_page)``. Pages outside ``pages`` are never
    scraped, which is what stops a full-corpus parse from running the expensive table
    detector over packaging, labelling and leaflet pages.
    """
    rows_by_page: dict[int, list[str]] = {}
    counts: dict[int, int] = {}
    for page_number in pages:
        if page_number < 1 or page_number > pdf.page_count:
            continue
        page = pdf[page_number - 1]
        try:
            finder = page.find_tables()
        except Exception:  # pragma: no cover - defensive: malformed page geometry
            counts[page_number] = 0
            continue
        counts[page_number] = len(finder.tables)
        rows: list[str] = []
        for table in finder.tables:
            rows.extend(_flatten_rows(table.extract()))
        rows_by_page[page_number] = rows
    return rows_by_page, counts


def _collect_lines(pdf: fitz.Document) -> tuple[list[Line], list[tuple[str, ...]]]:
    """Extract text lines with font facts, per page. No table work here.

    Table detection is deliberately not part of this pass: it is the expensive part of
    parsing a large PDF and only the pages a section owns ever need it.
    """
    lines: list[Line] = []
    pages: list[tuple[str, ...]] = []
    for page_index in range(pdf.page_count):
        page = pdf[page_index]
        page_number = page_index + 1
        page_line_texts: list[str] = []
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for raw_line in block["lines"]:
                spans = raw_line.get("spans", [])
                if not spans:
                    continue
                text = _normalise("".join(span["text"] for span in spans))
                if not text:
                    continue
                bold = any(bool(span.get("flags", 0) & _BOLD_FLAG) for span in spans)
                size = max(float(span.get("size", 0.0)) for span in spans)
                lines.append(Line(text=text, page=page_number, bold=bold, size=size))
                page_line_texts.append(text)
        pages.append(tuple(page_line_texts))
    return lines, pages


def _heading_at(lines: list[Line], index: int) -> tuple[str, str, int] | None:
    """Return ``(number, title, first_body_index)`` when a heading starts here."""
    line = lines[index]
    if not line.bold:
        return None
    if _DOTTED_LEADER_RE.search(line.text):
        return None

    combined = _NUMBER_TITLE_RE.match(line.text)
    if combined is not None:
        number, title = combined.group(1), combined.group(2)
        if len(title) <= _MAX_TITLE_CHARS:
            return number, title, index + 1
        return None

    number_only = _NUMBER_LINE_RE.match(line.text)
    if number_only is None:
        return None
    number = number_only.group(1)
    following = index + 1
    while following < len(lines) and not lines[following].text:
        following += 1
    if following >= len(lines) or not lines[following].bold:
        return None
    title = lines[following].text
    if len(title) > _MAX_TITLE_CHARS:
        return None
    return number, title, following + 1


def _contains_marker(lines: list[Line], markers: tuple[str, ...], start: int = 0) -> int | None:
    """Index of the first line containing any marker anywhere in its text.

    Used for the leaflet start, where the marker shares a line with other text,
    e.g. Greek ``ΕΠΙΣΗΜΑΝΣΗ ΚΑΙ ΦΥΛΛΟ ΟΔΗΓΙΩΝ ΧΡΗΣHΣ``.
    """
    wanted = tuple(m.upper() for m in markers)
    for index in range(start, len(lines)):
        text = lines[index].text.strip().upper()
        if not text:
            continue
        if any(marker in text for marker in wanted):
            return index
    return None


def _marker_index(lines: list[Line], markers: tuple[str, ...], start: int = 0) -> int | None:
    """Index of the first line equal to or starting with any marker."""
    wanted = tuple(m.upper() for m in markers)
    for index in range(start, len(lines)):
        text = lines[index].text.strip().upper()
        if not text:
            continue
        for marker in wanted:
            if text == marker or text.startswith(marker):
                return index
    return None


def _section_spans(
    lines: list[Line],
    wanted: tuple[str, ...],
    headings_vocabulary: tuple[str, ...],
) -> dict[str, tuple[str, int, int]]:
    """Map each wanted number to ``(title, body_start, body_end)``.

    Boundaries come from the next heading in ``headings_vocabulary``, so ``6.4``
    stops at ``6.5``. A repeated number later in the file (a second strength's
    SmPC, a repeated leaflet) is ignored: first occurrence wins, so the
    numbering a reader expects is what gets stored.
    """
    headings: list[tuple[int, str, str, int]] = []
    for index in range(len(lines)):
        heading = _heading_at(lines, index)
        if heading is None:
            continue
        number, title, body_start = heading
        if number in headings_vocabulary:
            headings.append((index, number, title, body_start))

    spans: dict[str, tuple[str, int, int]] = {}
    for position, (_index, number, title, body_start) in enumerate(headings):
        if number not in wanted or number in spans:
            continue
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        if end <= body_start:
            continue
        spans[number] = (title, body_start, end)
    return spans


def _table_rows_in_range(
    lines: list[Line],
    start: int,
    end: int,
    tables_by_page: dict[int, list[str]],
) -> list[str]:
    """Flattened table rows from the pages this section spans."""
    if start >= len(lines):
        return []
    first_page = lines[start].page
    last_page = lines[end - 1].page if end - 1 < len(lines) else lines[-1].page
    rows: list[str] = []
    for page in range(first_page, last_page + 1):
        rows.extend(tables_by_page.get(page, ()))
    return rows


def _smpc_window_end(lines: list[Line]) -> int:
    """Index where the SmPC (Annex I) body ends: the next annex marker, or EOF."""
    for markers in (_ANNEX_II_MARKERS, _ANNEX_III_MARKERS):
        index = _marker_index(lines, markers)
        if index is not None:
            return index
    return len(lines)


def parse_document(path: Path, *, doc: str, lang: str) -> ParsedDocument:
    """Parse one product-information PDF into numbered sections."""
    if doc not in {"smpc", "leaflet"}:
        msg = f"unsupported document kind: {doc}"
        raise ValueError(msg)
    if lang not in {"en", "el"}:
        msg = f"unsupported language: {lang}"
        raise ValueError(msg)

    with fitz.open(path) as pdf:
        if pdf.page_count == 0:
            msg = f"{path}: PDF has no pages"
            raise ValueError(msg)
        lines, pages = _collect_lines(pdf)
        page_count = pdf.page_count
        window = lines
        spans: dict[str, tuple[str, int, int]] = {}
        if doc == "smpc":
            window_end = _smpc_window_end(lines)
            window = lines[:window_end]
            spans = _section_spans(window, SMPC_WANTED, SMPC_HEADINGS)
        # Table detection is the expensive part of parsing. Only the 4.8 span can receive
        # flattened side-effect rows, so no other page is handed to the detector.
        target_pages: list[int] = []
        if doc == "smpc" and "4.8" in spans:
            _title, body_start, body_end = spans["4.8"]
            target_pages = sorted({line.page for line in window[body_start:body_end]})
        tables_by_page, table_counts = _tables_for_pages(pdf, target_pages)

    if not lines:
        msg = f"{path}: no extractable text (scanned PDF?)"
        raise ValueError(msg)

    if doc == "smpc":
        wanted = SMPC_WANTED
        vocabulary = SMPC_HEADINGS
    else:
        wanted = LEAFLET_WANTED
        vocabulary = LEAFLET_HEADINGS
        # The leaflet is Annex IIIB: skip the labelling annex (Annex IIIA) and
        # start at the "Package leaflet" marker after Annex III.
        annex = _contains_marker(lines, _LEAFLET_START_MARKERS)
        start = annex if annex is not None else 0
        leaflet = _contains_marker(lines, _LEAFLET_MARKERS, start=start + 1)
        window = lines[leaflet if leaflet is not None else start :]
        spans = _section_spans(window, wanted, vocabulary)

    sections: list[Section] = []
    for number in wanted:
        span = spans.get(number)
        if span is None:
            continue
        title, body_start, body_end = span
        body_lines = window[body_start:body_end]
        page = body_lines[0].page if body_lines else window[body_start - 1].page
        parts: list[str] = []
        body_text = re.sub(r"\n{3,}", "\n\n", "\n".join(line.text for line in body_lines)).strip()
        if body_text:
            parts.append(body_text)
        if doc == "smpc" and number == "4.8":
            rows = _table_rows_in_range(window, body_start, body_end, tables_by_page)
            if rows:
                parts.append("\n".join(rows))
        sections.append(
            Section(doc=doc, section=number, title=title, text="\n\n".join(parts), page=page)
        )

    return ParsedDocument(
        doc=doc,
        lang=lang,
        sections=tuple(sections),
        page_count=page_count,
        page_table_counts=dict(table_counts),
        _pages=tuple(pages),
    )


def split_section(section: Section, max_tokens: int = MAX_SECTION_TOKENS) -> list[Section]:
    """Split an oversized section at paragraph boundaries (SPEC section 8)."""
    text = section.text
    if len(text) // 4 <= max_tokens:
        return [section]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [section]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    budget = max_tokens * 4
    for paragraph in paragraphs:
        if current and current_len + len(paragraph) > budget:
            parts.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph)
    if current:
        parts.append("\n\n".join(current))
    return [
        Section(
            doc=section.doc,
            section=section.section,
            title=section.title,
            text=part,
            page=section.page,
        )
        for part in (parts or [text])
    ]


def parse_all_variants(path: Path, *, lang: str) -> tuple[ParsedDocument, ParsedDocument]:
    """Parse the SmPC and the leaflet from one Annex PDF.

    EMA publishes both documents in a single PDF, so both are read from the
    same bytes.
    """
    return (
        parse_document(path, doc="smpc", lang=lang),
        parse_document(path, doc="leaflet", lang=lang),
    )
