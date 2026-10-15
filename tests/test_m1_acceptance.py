"""M1 acceptance checks: section coverage and readable side-effect tables.

Fast unit tests run in every suite. The real-corpus checks are opt-in
(``MEDLEAF_M1_CORPUS=1``) so a normal run never needs the downloaded PDFs:

    MEDLEAF_M1_CORPUS=1 PYTHONPATH=src python -m pytest tests/test_m1_acceptance.py -q

They read local assets only (``data/raw/<slug>/<slug>_<lang>.pdf``); no network.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from medleaf.config import get_settings
from medleaf.ingest.chunk import (
    build_sections,
    flatten_table_rows,
    is_readable_frequency_row,
)
from medleaf.ingest.pipeline import parse_pdf
from medleaf.ingest.store import load_medicines
from medleaf.models import DocKind, Lang
from medleaf.sections import LEAFLET_REQUIRED, SMPC_REQUIRED

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Table flattener: the shapes real EMA files use -------------------------------

# Xarelto-class grid: the header row is frequency bands, and there is no reaction
# column. Column position carries the frequency.
_GRID_ROWS = [
    ["Common", "Uncommon", "Rare", "Very rare", "Not known"],
    ["Blood and lymphatic system disorders", "", "", "", ""],
    ["Anaemia (incl. respective lab)", "Thrombocytosis (incl. platelet)", "", "", ""],
    ["Immune system disorders", "", "", "", ""],
    ["", "Allergic reaction, dermatitis", "", "Anaphylactic reactions", ""],
]


def test_grid_table_keeps_the_column_frequency() -> None:
    """'Anaemia' is Common and 'Thrombocytosis' Uncommon, not one glued reaction."""
    lines = flatten_table_rows(_GRID_ROWS)

    assert "Anaemia (incl. respective lab) - Common" in lines
    assert "Thrombocytosis (incl. platelet) - Uncommon" in lines
    # The old bug flattened these two into "Anaemia ... - Thrombocytosis ...".
    assert not any("Anaemia" in line and "Thrombocytosis" in line for line in lines)
    # The system-organ-class heading is kept, and a continuation row still resolves
    # its column frequency even though the first cell is empty.
    assert "Blood and lymphatic system disorders" in lines
    assert "Allergic reaction, dermatitis - Uncommon" in lines


def test_grid_first_column_only_reaction_keeps_its_frequency() -> None:
    """A non-bold first-column-only row is a Common reaction, not a heading.

    Font weight is the only signal that separates it from a bold SOC heading such as
    ``Eye disorders``; without it the frequency was silently dropped.
    """
    rows = [
        ["Common", "Uncommon", "Rare"],
        ["Eye disorders", "", ""],
        ["Eye haemorrhage (incl. conjunctival haemorrhage)", "", ""],
        ["Hypotension, haematoma", "", ""],
    ]
    bolds: list[list[bool | None]] = [
        [True, True, True],
        [True, None, None],
        [False, False, False],
        [False, False, False],
    ]

    lines = flatten_table_rows(rows, bolds)

    # The bold SOC heading stays a bare heading and never takes a frequency.
    assert "Eye disorders" in lines
    assert not any(line.startswith("Eye disorders") and " - " in line for line in lines)
    # The regular first-column-only rows become reactions with column 0's frequency.
    assert "Eye haemorrhage (incl. conjunctival haemorrhage) - Common" in lines
    assert "Hypotension, haematoma - Common" in lines


def test_grid_first_column_only_defaults_to_heading_without_font_metadata() -> None:
    """With no font metadata an ambiguous row stays a heading (safe, no wrong band)."""
    rows = [["Common", "Uncommon"], ["Ambisomething", ""], ["Anaemia", "Thrombocytosis"]]

    lines = flatten_table_rows(rows)

    assert "Ambisomething" in lines
    assert not any(line.startswith("Ambisomething") and " - " in line for line in lines)
    assert "Anaemia - Common" in lines
    assert "Thrombocytosis - Uncommon" in lines


def test_labelled_table_uses_named_frequency_columns() -> None:
    """The Reaction|Frequency shape keeps the named bands, as Humira uses."""
    rows = [
        ["", "Frequency", "Adverse Reaction"],
        ["Infections and infestations", "Very common", "Respiratory tract infections"],
        ["", "Common", "Skin cancer"],
    ]
    lines = flatten_table_rows(rows)

    # Continuation row inherits the SOC label above it.
    assert (
        "Infections and infestations - Frequency: Very common; Adverse Reaction: "
        "Respiratory tract infections" in lines
    )
    assert lines[1].startswith("Infections and infestations - ")
    assert any("Skin cancer" in line for line in lines)


# --- Section coverage is measured per document section, not per chunk part ---------


def test_build_sections_keep_the_base_number_but_unique_chunk_ids() -> None:
    """A long section splits into parts; the section key stays '4.8' for coverage."""
    text = "\n\n".join(f"paragraph {index} " + "word " * 60 for index in range(4))
    sections = build_sections(
        medicine="demo",
        lang=Lang.EN,
        doc=DocKind.SMPC,
        raw_sections=[("4.8", "Undesirable effects", text)],
        source_url="https://example.invalid/x.pdf",
        revision_date="2026-01-01",
        corpus_version="test",
        max_tokens=100,
    )

    assert len(sections) > 1
    assert {section.section for section in sections} == {"4.8"}
    assert len({section.chunk_id for section in sections}) == len(sections)


def test_coverage_predicate_is_falsifiable() -> None:
    """A missing required section must make the coverage row fail."""
    from medleaf.cli import CoverageRow

    complete = CoverageRow("demo", "en", 28, 6, [], [])
    assert complete.ok is True

    broken = CoverageRow("demo", "en", 27, 6, ["4.8"], [])
    assert broken.ok is False
    assert broken.missing_smpc == ["4.8"]


# --- Opt-in real-corpus acceptance ------------------------------------------------


def _corpus_medicines():  # type: ignore[no-untyped-def]
    settings = get_settings()
    return settings, load_medicines(settings.medicines_file)


def _require_corpus() -> None:
    if os.environ.get("MEDLEAF_M1_CORPUS") != "1":
        pytest.skip("set MEDLEAF_M1_CORPUS=1 to run the real-corpus acceptance checks")
    # Opted in: missing assets are a failure, never a green skip.
    settings = get_settings()
    if not settings.medicines_file.exists():
        pytest.fail("MEDLEAF_M1_CORPUS=1 but data/medicines.yaml is missing; run `medleaf resolve`")
    for medicine in load_medicines(settings.medicines_file):
        for lang in (Lang.EN, Lang.EL):
            path = settings.pdf_path(medicine.slug, lang.value)
            if not path.exists():
                pytest.fail(f"MEDLEAF_M1_CORPUS=1 but missing corpus PDF: {path}")


def _parse(medicine, lang):  # type: ignore[no-untyped-def]
    settings = get_settings()
    return parse_pdf(
        medicine=medicine,
        lang=lang,
        pdf_path=settings.pdf_path(medicine.slug, lang.value),
        corpus_version=settings.corpus_version,
        max_tokens=settings.max_section_tokens,
    )


@pytest.mark.parametrize("slug", ["eliquis", "humira", "jardiance", "ozempic", "xarelto"])
@pytest.mark.parametrize("lang", [Lang.EN, Lang.EL])
def test_corpus_section_coverage_and_tables(slug: str, lang: Lang) -> None:
    """Every required SmPC/leaflet section is present and 4.8 has frequency rows."""
    _require_corpus()
    _, medicines = _corpus_medicines()
    medicine = next(item for item in medicines if item.slug == slug)
    parsed = _parse(medicine, lang)
    smpc = {section.section for section in parsed.by_doc(DocKind.SMPC)}
    leaflet = {section.section for section in parsed.by_doc(DocKind.LEAFLET)}

    assert not [key for key in SMPC_REQUIRED if key not in smpc], f"{slug}/{lang}: {sorted(smpc)}"
    assert not [key for key in LEAFLET_REQUIRED if key not in leaflet], f"{slug}/{lang}"

    rows = [
        line
        for section in parsed.by_doc(DocKind.SMPC)
        if section.section == "4.8"
        for line in section.text.splitlines()
        if is_readable_frequency_row(line)
    ]
    assert rows, f"{slug}/{lang}: no readable 4.8 frequency rows"


def test_xarelto_en_frequency_associations() -> None:
    _require_corpus()
    _, medicines = _corpus_medicines()
    medicine = next(item for item in medicines if item.slug == "xarelto")
    parsed = _parse(medicine, Lang.EN)
    text = "\n".join(
        section.text for section in parsed.by_doc(DocKind.SMPC) if section.section == "4.8"
    )
    lines = text.splitlines()

    anaemia = [line for line in lines if line.startswith("Anaemia")]
    thrombo = [line for line in lines if "Thrombocytosis" in line]
    assert anaemia, "Anaemia row missing from xarelto_en 4.8"
    assert any("Common" in line for line in anaemia), anaemia
    assert thrombo, "Thrombocytosis row missing from xarelto_en 4.8"
    assert any("Uncommon" in line for line in thrombo), thrombo
    # The old bug glued two reactions of different frequencies into one row; that row
    # would start with 'Anaemia' and carry 'Thrombocytosis' in the same line.
    glued = [line for line in lines if line.startswith("Anaemia") and "Thrombocytosis" in line]
    assert not glued, glued


def test_xarelto_en_first_column_only_reaction_keeps_frequency() -> None:
    """Real-PDF evidence: a first-column-only reaction is Common, not a dropped heading.

    Xarelto's 4.8 table is grid-shaped (``Common | Uncommon | ...``). Rows such as
    ``Eye haemorrhage (incl. conjunctival haemorrhage)`` and ``Hypotension, haematoma``
    appear only in the Common column; they are regular weight, while the surrounding
    SOC headings are bold and must stay bare.
    """
    _require_corpus()
    _, medicines = _corpus_medicines()
    medicine = next(item for item in medicines if item.slug == "xarelto")
    parsed = _parse(medicine, Lang.EN)
    text = "\n".join(
        section.text for section in parsed.by_doc(DocKind.SMPC) if section.section == "4.8"
    )
    lines = text.splitlines()

    eye = [line for line in lines if line.startswith("Eye haemorrhage")]
    assert eye, "Eye haemorrhage row missing from xarelto_en 4.8"
    assert any("Common" in line for line in eye), eye

    hypoten = [line for line in lines if line.startswith("Hypotension")]
    assert hypoten, "Hypotension row missing from xarelto_en 4.8"
    assert any("Common" in line for line in hypoten), hypoten

    # Bold SOC headings must never be given a frequency band. (A heading line may be
    # dropped by de-duplication against surrounding prose, but it must never acquire
    # an adverse-reaction frequency.)
    for heading in ("Eye disorders", "Nervous system disorders", "Vascular disorders"):
        mismatched = [
            line
            for line in lines
            if line.startswith(heading) and any(freq in line for freq in (" - Common", " - Rare"))
        ]
        assert not mismatched, f"SOC heading given a frequency: {mismatched}"
