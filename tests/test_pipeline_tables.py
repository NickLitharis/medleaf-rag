"""Regression tests for table-extraction scope and cost on large PDFs.

PyMuPDF's table detector is the expensive part of parsing an EMA product-information
PDF, and it used to run on every page. These tests pin the fixed scope — only the pages
of an actual SmPC 4.8 span — with a tiny synthetic PDF and a call counter, so they stay
fast and never need the corpus.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from medleaf import parse as legacy_parse
from medleaf.ingest import pipeline
from medleaf.ingest.pdf import RawLine


class _FakeTable:
    """Stand-in for a PyMuPDF table with scripted rows."""

    def __init__(self, rows: list[list[str]]) -> None:
        self._rows = rows

    def extract(self) -> list[list[str]]:
        return self._rows


class _FakeFinder:
    """Stand-in for the object returned by ``Page.find_tables``."""

    def __init__(self, tables: list[_FakeTable]) -> None:
        self.tables = tables


def _make_pdf(tmp_path: Path, pages: int = 3) -> str:
    """Write a tiny blank synthetic PDF and return its path."""
    document = fitz.open()
    for _ in range(pages):
        document.new_page(width=595, height=842)
    path = tmp_path / "synthetic.pdf"
    document.save(str(path))
    document.close()
    return str(path)


def _count_calls(
    monkeypatch: pytest.MonkeyPatch, rows: list[list[str]] | None = None
) -> list[int]:
    """Patch ``Page.find_tables`` to record 1-based pages and return scripted tables."""
    scanned: list[int] = []

    def fake_find_tables(page: fitz.Page, *args: object, **kwargs: object) -> _FakeFinder:
        scanned.append(page.number + 1)
        if rows is not None and page.number == 0:
            return _FakeFinder([_FakeTable(rows)])
        return _FakeFinder([])

    monkeypatch.setattr(fitz.Page, "find_tables", fake_find_tables)
    return scanned


def _texts_by_section(result: list[tuple[str, str, str]]) -> dict[str, str]:
    return {number: text for number, _title, text in result}


def test_detector_stops_at_the_section_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4.9's pages are not attributed to 4.8 once 4.9's heading has started."""
    pdf_path = _make_pdf(tmp_path, pages=4)
    scanned = _count_calls(monkeypatch)
    lines = [
        RawLine(text="4.8 Undesirable effects", y=10.0, x=10.0, page=1),
        RawLine(text="body of 4.8", y=20.0, x=10.0, page=2),
        RawLine(text="4.9 Overdose", y=10.0, x=10.0, page=3),
        RawLine(text="body of 4.9", y=20.0, x=10.0, page=4),
    ]

    pipeline._attach_tables(
        pdf_path,
        [("4.8", "Undesirable effects", "body"), ("4.9", "Overdose", "overdose body")],
        lines,
    )

    assert scanned == [1, 2]


def test_detector_never_reaches_the_leaflet_annex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Annex II ends the SmPC, so packaging and leaflet pages are never scanned."""
    pdf_path = _make_pdf(tmp_path, pages=5)
    scanned = _count_calls(monkeypatch)
    lines = [
        RawLine(text="4.8 Undesirable effects", y=10.0, x=10.0, page=1),
        RawLine(text="body of 4.8", y=20.0, x=10.0, page=2),
        RawLine(text="ANNEX II", y=10.0, x=10.0, page=3),
        RawLine(text="packaging text", y=10.0, x=10.0, page=4),
        RawLine(text="B. PACKAGE LEAFLET", y=10.0, x=10.0, page=5),
    ]

    pipeline._attach_tables(pdf_path, [("4.8", "Undesirable effects", "body")], lines)

    assert scanned == [1, 2]


def test_heading_page_is_owned_even_when_the_heading_starts_mid_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 4.8 heading below 4.7 body on the same page still owns that page's table."""
    pdf_path = _make_pdf(tmp_path, pages=2)
    _count_calls(monkeypatch, rows=[["Reaction", "Frequency"], ["Nausea", "Common"]])
    lines = [
        RawLine(text="4.7 Effects on ability to drive", y=10.0, x=10.0, page=1),
        RawLine(text="4.8 Undesirable effects", y=200.0, x=10.0, page=1),
        RawLine(text="body of 4.8", y=20.0, x=10.0, page=2),
    ]

    result = pipeline._attach_tables(
        pdf_path,
        [
            ("4.7", "Effects on ability to drive", "drive body"),
            ("4.8", "Undesirable effects", "body"),
        ],
        lines,
    )

    assert set(_texts_by_section(result)) == {"4.7", "4.8"}
    # The heading page is scanned, and its table is attributed to 4.8, not 4.7.
    assert _texts_by_section(result)["4.8"] == "body\n\nNausea - Common"
    assert _texts_by_section(result)["4.7"] == "drive body"


def test_detector_does_not_scan_other_sections_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pages owned by later SmPC sections such as 6.4 are not table-scanned."""
    pdf_path = _make_pdf(tmp_path, pages=4)
    scanned = _count_calls(monkeypatch)
    lines = [
        RawLine(text="4.8 Undesirable effects", y=10.0, x=10.0, page=1),
        RawLine(text="4.9 Overdose", y=10.0, x=10.0, page=2),
        RawLine(text="6.4 Special precautions for storage", y=10.0, x=10.0, page=3),
        RawLine(text="store below 25 C", y=10.0, x=10.0, page=4),
    ]

    pipeline._attach_tables(
        pdf_path,
        [
            ("4.8", "Undesirable effects", "body"),
            ("6.4", "Special precautions for storage", "store below 25 C"),
        ],
        lines,
    )

    assert scanned == [1]


def test_detector_runs_once_per_target_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One detector call per 4.8 page: no page is reopened or re-scraped."""
    pdf_path = _make_pdf(tmp_path, pages=3)
    scanned = _count_calls(monkeypatch)
    lines = [
        RawLine(text="4.8 Undesirable effects", y=10.0, x=10.0, page=1),
        RawLine(text="body text", y=20.0, x=10.0, page=2),
        RawLine(text="4.9 Overdose", y=10.0, x=10.0, page=3),
    ]

    pipeline._attach_tables(pdf_path, [("4.8", "Undesirable effects", "body")], lines)

    assert scanned == [1, 2]


def test_detector_skips_when_4_8_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No 4.8 section means no target page, so the detector is never called."""
    pdf_path = _make_pdf(tmp_path, pages=3)
    scanned = _count_calls(monkeypatch)
    lines = [
        RawLine(text="6.4 Special precautions for storage", y=10.0, x=10.0, page=1),
        RawLine(text="store below 25 C", y=20.0, x=10.0, page=2),
    ]

    result = pipeline._attach_tables(
        pdf_path, [("6.4", "Special precautions for storage", "store below 25 C")], lines
    )

    assert scanned == []
    assert result == [("6.4", "Special precautions for storage", "store below 25 C")]


def test_detector_skips_when_there_are_no_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = _make_pdf(tmp_path, pages=2)
    scanned = _count_calls(monkeypatch)

    result = pipeline._attach_tables(pdf_path, [], [RawLine(text="x", y=10.0, x=10.0, page=1)])

    assert result == []
    assert scanned == []


def test_legacy_parser_detects_tables_once_per_requested_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``medleaf.parse`` scrapes only the requested 4.8 pages, once each."""
    pdf_path = _make_pdf(tmp_path, pages=3)
    scanned = _count_calls(monkeypatch)
    with fitz.open(pdf_path) as document:
        rows_by_page, counts = legacy_parse._tables_for_pages(document, [2])

    assert scanned == [2]
    assert counts == {2: 0}
    assert rows_by_page == {2: []}
