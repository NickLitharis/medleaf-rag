"""Boundary regression tests for the EMA product-information parser.

These pin the failures that made the served index unusable: a product-information PDF
can contain several complete SmPCs (one per presentation) with the *same* section
numbers, and the package leaflet starts with a numbered contents list. Taking every
heading produced one giant last section (``smpc:10`` up to hundreds of kilobytes) that
swallowed the annex text and every later copy, which then dominated sparse retrieval.

The synthetic cases run everywhere. The real-corpus coverage check is opt-in with
``MEDLEAF_M1_CORPUS=1`` because it needs the downloaded PDFs.
"""

from __future__ import annotations

import os

import pytest

from medleaf.config import get_settings
from medleaf.ingest.pdf import RawLine, extract_lines, first_copy_headings, split_sections
from medleaf.ingest.store import load_medicines
from medleaf.models import DocKind, Lang
from medleaf.sections import LEAFLET_REQUIRED, SMPC_REQUIRED


def _lines(texts: list[str]) -> list[RawLine]:
    """RawLines with increasing baselines, one per text (page 1)."""
    return [
        RawLine(text=text, y=float(index * 12), x=0.0, page=1)
        for index, text in enumerate(texts)
    ]


def _keys(sections: list[tuple[str, str, str]]) -> list[str]:
    return [key for key, _, _ in sections]


def _text(sections: list[tuple[str, str, str]], key: str) -> str:
    return "\n".join(text for number, _, text in sections if number == key)


_SMPC_TWO_COPIES = [
    "ANNEX I",
    "1. NAME OF THE MEDICINAL PRODUCT",
    "first copy name",
    "first copy name continued",
    "2. QUALITATIVE AND QUANTITATIVE COMPOSITION",
    "first copy composition",
    "first copy composition continued",
    "10. DATE OF REVISION OF THE TEXT",
    "first copy revision",
    "first copy revision continued",
    # Second presentation: the same sections start again.
    "1. NAME OF THE MEDICINAL PRODUCT",
    "second copy name",
    "second copy name continued",
    "2. QUALITATIVE AND QUANTITATIVE COMPOSITION",
    "second copy composition",
]


def test_first_copy_stops_before_the_next_presentation() -> None:
    """A repeated section 1 + 2 restarts the document; only the first copy is kept."""
    sections = split_sections(_lines(_SMPC_TWO_COPIES), doc=DocKind.SMPC)

    assert _keys(sections) == ["1", "2", "10"]
    assert "first copy name" in _text(sections, "1")
    assert "second copy name" not in _text(sections, "1")
    assert "second copy composition" not in _text(sections, "2")


def test_leaflet_contents_list_is_not_body() -> None:
    """A numbered contents list at the start of a leaflet is dropped, not indexed."""
    lines = _lines(
        [
            "B. PACKAGE LEAFLET",
            "1. What demo is and what it is used for",
            "2. What you need to know before you take demo",
            "3. How to take demo",
            "4. Possible side effects",
            "5. How to store demo",
            "6. Contents of the pack and other information",
            "1. What demo is and what it is used for",
            "real body one",
            "real body one continued",
            "2. What you need to know before you take demo",
            "real body two",
            "real body two continued",
        ]
    )
    sections = split_sections(lines, doc=DocKind.LEAFLET)

    assert "real body one" in _text(sections, "1")
    assert "real body two" in _text(sections, "2")


def test_numbered_instruction_step_does_not_end_the_copy() -> None:
    """A body step numbered ``1.`` is not a new document copy."""
    lines = _lines(
        [
            "1. What demo is and what it is used for",
            "body one",
            "body one continued",
            "2. What you need to know before you take demo",
            "body two",
            "body two continued",
            "3. How to take demo",
            "body three",
            "body three continued",
            "1. How to puncture the foil",
            "step body",
            "step body continued",
            "4. Possible side effects",
            "body four",
            "body four continued",
        ]
    )
    sections = split_sections(lines, doc=DocKind.LEAFLET)

    assert _keys(sections) == ["1", "2", "3", "4"]
    assert "body four" in _text(sections, "4")
    assert "step body" in _text(sections, "3")


def test_no_annex_marker_falls_back_to_the_end_of_the_file() -> None:
    """With no annex marker and no later copy, the last section runs to EOF."""
    lines = _lines(
        [
            "1. NAME OF THE MEDICINAL PRODUCT",
            "only copy name",
            "only copy name continued",
            "10. DATE OF REVISION OF THE TEXT",
            "trailing revision text",
            "trailing revision text continued",
        ]
    )
    hits, copy_end = first_copy_headings(lines, doc=DocKind.SMPC)
    sections = split_sections(lines, doc=DocKind.SMPC)

    assert copy_end == len(lines)
    assert [hit.key for hit in hits] == ["1", "10"]
    assert "trailing revision text" in _text(sections, "10")


@pytest.mark.skipif(
    os.environ.get("MEDLEAF_M1_CORPUS") != "1",
    reason="set MEDLEAF_M1_CORPUS=1 to run the real-corpus coverage check",
)
def test_real_corpus_has_every_required_section_in_the_first_copy() -> None:
    """All 40 medicines keep every required SmPC/leaflet section after the boundary fix."""
    settings = get_settings()
    medicines = load_medicines(settings.medicines_file)
    assert len(medicines) == 40, f"expected the 40-medicine corpus, got {len(medicines)}"

    missing: list[str] = []
    for medicine in medicines:
        for lang in (Lang.EN, Lang.EL):
            path = settings.pdf_path(medicine.slug, lang.value)
            if not path.exists():
                pytest.fail(f"MEDLEAF_M1_CORPUS=1 but missing corpus PDF: {path}")
            lines = extract_lines(str(path))
            smpc = {key for key, _, _ in split_sections(lines, doc=DocKind.SMPC)}
            leaflet = {key for key, _, _ in split_sections(lines, doc=DocKind.LEAFLET)}
            for key in SMPC_REQUIRED:
                if key not in smpc:
                    missing.append(f"{medicine.slug}/{lang.value}/smpc:{key}")
            for key in LEAFLET_REQUIRED:
                if key not in leaflet:
                    missing.append(f"{medicine.slug}/{lang.value}/leaflet:{key}")
    assert not missing, f"required sections missing: {missing}"
