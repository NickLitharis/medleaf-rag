"""Bounded real-PDF smoke test for both parsers. Skipped by default.

The check is opt-in so a normal test run never pays for a full 300-page parse. Run it
with a hard external timeout and a real file from the local corpus, e.g.::

    MEDLEAF_SMOKE_PDF=.tmp/raw/xarelto_el.pdf timeout 300 \\
        python -m pytest tests/test_parser_perf_smoke.py -q -s

It asserts that both implementations still find SmPC 4.8, that table detection ran, and
that it ran on far fewer pages than the file has.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import fitz
import pytest
from pydantic import HttpUrl

from medleaf import parse as legacy_parse
from medleaf.ingest.chunk import is_frequency_line
from medleaf.ingest.pipeline import parse_pdf
from medleaf.models import DocKind, DocumentRef, Lang, Medicine

# Generous upper bound for one 370-page Greek PDF on a dev laptop; the shell timeout is
# the hard stop, this is the regression guard.
_MAX_SECONDS = 240.0


def _smoke_medicine() -> Medicine:
    url = HttpUrl("https://www.ema.europa.eu/")
    return Medicine(
        slug="smoke",
        name="Smoke",
        active_substance="n/a",
        epar_url=url,
        documents={
            Lang.EN: DocumentRef(lang=Lang.EN, url=url),
            Lang.EL: DocumentRef(lang=Lang.EL, url=url),
        },
    )


def test_real_large_pdf_parses_within_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = os.environ.get("MEDLEAF_SMOKE_PDF") or ""
    if not configured:
        pytest.skip("MEDLEAF_SMOKE_PDF not set")
    path = Path(configured)
    if not path.exists():
        pytest.skip(f"missing smoke PDF: {path}")

    original_find_tables = fitz.Page.find_tables
    scanned: list[int] = []

    def counting_find_tables(page: fitz.Page, *args: object, **kwargs: object) -> object:
        scanned.append(page.number + 1)
        return original_find_tables(page, *args, **kwargs)

    monkeypatch.setattr(fitz.Page, "find_tables", counting_find_tables)

    lang = Lang.EL if path.stem.endswith("_el") else Lang.EN

    started = time.perf_counter()
    parsed = parse_pdf(
        medicine=_smoke_medicine(),
        lang=lang,
        pdf_path=path,
        corpus_version="smoke",
        max_tokens=400,
    )
    elapsed = time.perf_counter() - started
    pipeline_detector_pages = len(scanned)

    smpc = parsed.by_doc(DocKind.SMPC)
    numbers = [section.section for section in smpc]
    parts_48 = [section for section in smpc if section.section.startswith("4.8")]
    frequency_lines = sum(
        1
        for section in parts_48
        for line in section.text.splitlines()
        if is_frequency_line(line)
    )
    print(
        f"\nsmoke: pipeline {path.name} parsed in {elapsed:.1f}s -> "
        f"{len(smpc)} SmPC sections, 4.8 parts={len(parts_48)}, "
        f"4.8 frequency lines={frequency_lines}, table-detector pages={pipeline_detector_pages}"
    )

    legacy_started = time.perf_counter()
    legacy = legacy_parse.parse_document(path, doc="smpc", lang=lang.value)
    legacy_elapsed = time.perf_counter() - legacy_started
    legacy_pages = legacy.pages_with_tables()
    print(
        f"smoke: legacy {path.name} parsed in {legacy_elapsed:.1f}s -> "
        f"4.8 present={legacy.section('4.8') is not None}, "
        f"pages_with_tables={len(legacy_pages)} of {legacy.page_count}"
    )

    assert any(number.startswith("4.1") for number in numbers), f"missing SmPC 4.1 in {numbers}"
    assert any(number.startswith("4.8") for number in numbers), f"missing SmPC 4.8 in {numbers}"
    assert pipeline_detector_pages, "pipeline table detection never ran"
    assert elapsed < _MAX_SECONDS, f"parse took {elapsed:.1f}s (budget {_MAX_SECONDS:.0f}s)"

    assert legacy.section("4.8") is not None
    assert legacy_pages, "legacy table detection never found a table page"
    assert len(legacy_pages) < legacy.page_count, "legacy detector scanned every page"
