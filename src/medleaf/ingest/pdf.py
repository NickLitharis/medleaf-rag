"""PDF text extraction and section splitting for EMA product-information PDFs.

Layout reality (verified against real EMA files, 2026-10-15):

* One product-information PDF holds several annexes: Annex I (SmPC), Annex II,
  Annex III (labelling + package leaflet). SPEC section 8 says to keep the SmPC and
  the leaflet and skip the annexes in between.
* Headings do not always start on one line. PyMuPDF renders ``4.8 Undesirable
  effects`` as two adjacent text lines on the same baseline: ``4.8 `` and
  ``Undesirable effects``. Lines are therefore joined per baseline before matching.
* Greek documents use the same numbering (``4.8`` etc.) with Greek titles, which is
  what makes language-independent chunk IDs possible.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

import fitz  # PyMuPDF

from medleaf.models import DocKind, Lang
from medleaf.sections import (
    LEAFLET_NUMBERS,
    SMPC_NUMBERS,
    looks_like_leaflet_heading,
    smpc_title,
)

# Section patterns. The title itself is not matched here: titles are language
# specific, so only the number and a following capital/letter are required.
# Top-level SmPC headings are written '4. CLINICAL PARTICULARS' (a period after the
# number), while sub-sections are '4.1 Therapeutic indications' (no period). Both
# forms must be accepted, so the period is optional.
SMPC_HEADING_RE = re.compile(r"^(?P<num>[1-9][0-9]?(?:\.[0-9]{1,2})?)\s*\.?\s+(?P<title>\S.*)$")
LEAFLET_HEADING_RE = re.compile(r"^(?P<num>[1-6])\.\s*(?P<title>\S.*)$")

# Annex markers. The trailing capital is matched loosely because EMA's Greek files use
# Greek capital iota ('ΠΑΡΑΡΤΗΜΑ Ι', U+0399), which is visually identical to Latin 'I'.
ANNEX_I_RE = re.compile(r"^\s*(ANNEX\s+I$|ΠΑΡΑΡΤΗΜΑ\s+[ΙI]$)")
ANNEX_II_RE = re.compile(r"^\s*(ANNEX\s+II$|ΠΑΡΑΡΤΗΜΑ\s+[ΙI]{2}$)")
ANNEX_III_RE = re.compile(r"^\s*(ANNEX\s+III$|ΠΑΡΑΡΤΗΜΑ\s+[ΙI]{3}$)")
# Annex/labelling/leaflet start markers. The leading letter is matched with both the
# Latin and the visually identical Greek capital, because EMA mixes them inside one
# file ('B. ΦΥΛΛΟ ...' uses a Latin B while the rest is Greek).
LEAFLET_START_RE = re.compile(
    r"^\s*[BΒ]\.\s*(PACKAGE\s+LEAFLET|ΦΥΛΛΟ\s+ΟΔΗΓΙΩΝ\s+ΧΡΗΣΗΣ)$",
    re.IGNORECASE,
)
LABELLING_START_RE = re.compile(
    r"^\s*[AΑ]\.\s*(LABELLING|ΕΠΙΣΗΜΑΝΣΗ)$",
    re.IGNORECASE,
)

# Running heads/footers that would otherwise land inside section text.
PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
PAGE_NUMBER_INLINE_RE = re.compile(r"^\s*\d{1,4}\s+(?=\S)")
SOFT_HYPHEN_RE = re.compile(r"(\w)-\s*\n\s*(\w)")

# A table of contents lists several section numbers a few lines apart. Body headings
# are separated by the section text between them, so a run this tight and this long
# (starting at the first section) is contents, not content.
_CONTENTS_GAP_LINES = 2
_CONTENTS_MIN_RUN = 4


@dataclass(frozen=True)
class RawLine:
    """One visual line of the PDF, with the baseline used for joining split headings."""

    text: str
    y: float
    x: float
    page: int


#: Two spans whose baselines are within this many points are one visual line. The
#: SmPC number and its title are sometimes laid out on baselines ~1.6pt apart, so a
#: tighter tolerance leaves the number stranded and rejects the heading.
_BASELINE_TOLERANCE = 2.5


def iter_lines(page: fitz.Page, page_number: int) -> Iterator[RawLine]:
    """Yield visual lines, merging spans that share a baseline.

    PyMuPDF emits ``4.8 `` and ``Undesirable effects`` as separate lines even though
    they sit on the same baseline, so lines within a small vertical tolerance are
    joined in reading order.
    """
    data = page.get_text("dict")
    collected: list[RawLine] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if not text.strip():
                continue
            bbox = line.get("bbox", (0.0, 0.0, 0.0, 0.0))
            collected.append(
                RawLine(text=text.strip(), y=round(float(bbox[1]), 1), x=float(bbox[0]),
                        page=page_number)
            )

    collected.sort(key=lambda item: (item.y, item.x))
    merged: list[RawLine] = []
    for item in collected:
        if merged and abs(merged[-1].y - item.y) <= _BASELINE_TOLERANCE:
            previous = merged[-1]
            merged[-1] = RawLine(
                text=f"{previous.text} {item.text}".strip(),
                y=previous.y,
                x=min(previous.x, item.x),
                page=previous.page,
            )
        else:
            merged.append(item)
    yield from merged


def extract_lines(pdf_path: str, *, max_pages: int | None = None) -> list[RawLine]:
    """Extract every visual line from a PDF, in page order."""
    lines: list[RawLine] = []
    with fitz.open(pdf_path) as document:
        page_count = (
            document.page_count if max_pages is None else min(max_pages, document.page_count)
        )
        for index in range(page_count):
            lines.extend(iter_lines(document[index], index + 1))
    return lines


def clean_line(text: str) -> str:
    """Normalise a line: collapse runs of spaces, drop standalone page numbers."""
    text = text.replace("\u00ad", "").replace("\xa0", " ")
    text = PAGE_NUMBER_INLINE_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# PyMuPDF sometimes emits a sub-section number with a space after the period
# ('4. 1 Therapeutic indications' instead of '4.1 Therapeutic indications').
# Normalising it here keeps the heading patterns simple.
SPACED_SUBNUMBER_RE = re.compile(r"^([1-9])[.\u00b7]\s+([0-9]{1,2})(?=\s|$)")


def normalize_heading_line(text: str) -> str:
    """Repair PDF-inserted spaces inside a SmPC sub-section number."""
    return SPACED_SUBNUMBER_RE.sub(lambda m: f"{m.group(1)}.{m.group(2)}", text)


def find_annex_boundaries(lines: list[RawLine]) -> dict[str, int]:
    """Locate the annex starts by index into ``lines``.

    Keys are ``annex1``, ``annex2``, ``annex3``, ``leaflet`` and ``labelling``.
    Missing keys mean the marker was not present in that document.
    """
    boundaries: dict[str, int] = {}
    for index, line in enumerate(lines):
        text = clean_line(line.text)
        if "annex1" not in boundaries and ANNEX_I_RE.match(text):
            boundaries["annex1"] = index
        elif "annex2" not in boundaries and ANNEX_II_RE.match(text):
            boundaries["annex2"] = index
        elif "annex3" not in boundaries and ANNEX_III_RE.match(text):
            boundaries["annex3"] = index
        if "labelling" not in boundaries and LABELLING_START_RE.match(text):
            boundaries["labelling"] = index
        if "leaflet" not in boundaries and LEAFLET_START_RE.match(text):
            boundaries["leaflet"] = index
    return boundaries


@dataclass(frozen=True)
class HeadingHit:
    """A detected numbered heading."""

    index: int
    key: str
    title: str


def _scan_headings(
    lines: list[RawLine],
    start: int,
    end: int,
    numbers: tuple[str, ...],
    pattern: re.Pattern[str],
    title_fn: object,
) -> list[HeadingHit]:
    """All valid numbered headings in reading order, including repeated copies.

    Duplicates are deliberately *not* dropped here: a repeated heading is how the next
    copy of a multi-presentation document announces itself, and the caller uses that
    to stop at the first complete copy.
    """
    hits: list[HeadingHit] = []
    for index in range(start, end):
        text = normalize_heading_line(clean_line(lines[index].text))
        if not text:
            continue
        match = pattern.match(text)
        if not match:
            continue
        number = match.group("num")
        if number not in numbers:
            continue
        title = match.group("title").strip()
        validate = title_fn  # callable(number, title) -> bool
        if not callable(validate):  # pragma: no cover - defensive
            continue
        if not validate(number, title):
            continue
        hits.append(HeadingHit(index=index, key=number, title=title))
    return hits


def _order_map(numbers: tuple[str, ...]) -> dict[str, int]:
    """Document order of each section number (``'10'`` sorts after ``'9'``)."""
    return {number: index for index, number in enumerate(numbers)}


def _keep_chain(chain: list[HeadingHit], order: dict[str, int]) -> list[HeadingHit]:
    """Drop a tight run of headings that is a table of contents, not body text.

    A leaflet prints its contents list as several section headings a line or two apart,
    immediately followed by the real first heading. Body headings are separated by their
    own text, so a tight run that starts at section ``1`` and holds four or more headings
    is contents. When the real body's first heading is glued to the end of that run (the
    run's last heading is section ``1`` again), that heading is kept; otherwise the whole
    run is dropped and the body's own headings are picked up afterwards.
    """
    starts_early = order[chain[0].key] <= 1
    if len(chain) >= _CONTENTS_MIN_RUN and starts_early:
        if order[chain[-1].key] == 0:
            return [chain[-1]]
        return []
    return chain


def _drop_contents_runs(hits: list[HeadingHit], order: dict[str, int]) -> list[HeadingHit]:
    """Remove contents-list runs, grouping headings that sit on adjacent lines."""
    if not hits:
        return []
    kept: list[HeadingHit] = []
    chain: list[HeadingHit] = [hits[0]]
    for hit in hits[1:]:
        if hit.index - chain[-1].index <= _CONTENTS_GAP_LINES:
            chain.append(hit)
        else:
            kept.extend(_keep_chain(chain, order))
            chain = [hit]
    kept.extend(_keep_chain(chain, order))
    return kept


def _starts_new_copy(hits: list[HeadingHit], position: int, order: dict[str, int]) -> bool:
    """True when this section ``1`` heading really restarts the document.

    A numbered step in the body (``1. How to puncture the foil``) also matches the
    section pattern. A real restart is followed by the next section numbers in order,
    which an instruction step is not.
    """
    following = [order[hit.key] for hit in hits[position + 1 : position + 3]]
    return following == [1, 2]


def first_copy_headings(
    lines: list[RawLine],
    *,
    doc: DocKind,
) -> tuple[list[HeadingHit], int]:
    """Headings of the first complete document copy, and where that copy ends.

    EMA ships several complete SmPCs (one per presentation) inside one Annex I, and
    several package leaflets after Annex III. Only the first copy is indexed: the later
    copies repeat the same clinical sections and, if merged, make the last matched
    heading (``10`` or leaflet ``6``) swallow the whole rest of the file.

    Within one copy the section numbers strictly increase, so the first heading whose
    number is not greater than the last accepted one starts the next copy and marks the
    end of the first. If no later copy exists, the copy ends at the annex/first-annex
    boundary (or EOF).
    """
    boundaries = find_annex_boundaries(lines)
    if doc is DocKind.SMPC:
        start = boundaries.get("annex1", 0)
        ends = [
            boundaries[key]
            for key in ("annex2", "annex3", "labelling", "leaflet")
            if key in boundaries
        ]
        hard_end = min([*ends, len(lines)])
        numbers = SMPC_NUMBERS
        pattern = SMPC_HEADING_RE
        title_fn: object = smpc_title
    else:
        start = boundaries.get("leaflet", boundaries.get("annex3", 0))
        hard_end = len(lines)
        numbers = LEAFLET_NUMBERS
        pattern = LEAFLET_HEADING_RE
        title_fn = lambda number, title: looks_like_leaflet_heading(title)  # noqa: E731

    order = _order_map(numbers)
    scanned = _scan_headings(lines, start, hard_end, numbers, pattern, title_fn)
    hits = _drop_contents_runs(scanned, order)

    accepted: list[HeadingHit] = []
    copy_end = hard_end
    last_rank = -1
    for position, hit in enumerate(hits):
        rank = order[hit.key]
        if accepted and rank <= last_rank:
            if rank == 0 and _starts_new_copy(hits, position, order):
                # A section 1 followed by 2 and 3 restarts the document: the first
                # copy ends here and later copies are not indexed.
                copy_end = hit.index
                break
            # A step numbered ``1.`` inside the body is not a restart; ignore it.
            continue
        accepted.append(hit)
        last_rank = rank
    return accepted, copy_end


def split_sections(
    lines: list[RawLine],
    *,
    doc: DocKind,
) -> list[tuple[str, str, str]]:
    """Return ``(key, title, text)`` triples for one document kind.

    Headings are matched by number, and accepted only when the following text looks
    like the expected title for that number in either corpus language. That check is
    what stops cross-references such as ``see section 4.8`` or table cells from being
    mistaken for headings. Only the first complete document copy is returned.
    """
    hits, copy_end = first_copy_headings(lines, doc=doc)
    if not hits:
        return []

    sections: list[tuple[str, str, str]] = []
    for position, hit in enumerate(hits):
        body_start = hit.index + 1
        body_end = hits[position + 1].index if position + 1 < len(hits) else copy_end
        body_lines = [clean_line(line.text) for line in lines[body_start:body_end]]
        text = "\n".join(line for line in body_lines if line)
        sections.append((hit.key, hit.title, text))
    return sections


def rebuild_paragraphs(text: str) -> str:
    """Join wrapped lines back into paragraphs, keeping blank-line separation."""
    text = SOFT_HYPHEN_RE.sub(r"\1\2", text)
    lines = text.split("\n")
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if not line.strip():
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            continue
        buffer.append(line.strip())
    if buffer:
        paragraphs.append(" ".join(buffer))
    return "\n\n".join(paragraphs)


def detect_language(text: str) -> Lang:
    """Classify document language by Greek script ratio.

    EMA ships some language URLs that silently redirect to English, so the detected
    language is checked against the requested one after download.
    """
    sample = text[:20000]
    greek = sum(1 for char in sample if "\u0386" <= char <= "\u03ce")
    letters = sum(1 for char in sample if char.isalpha())
    if letters and greek / letters > 0.20:
        return Lang.EL
    return Lang.EN
