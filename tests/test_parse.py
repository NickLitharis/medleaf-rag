"""Regression tests for the section parser's language-independent rules.

These run without PDF I/O: they drive the heading logic with synthetic lines,
so they stay fast and pin the exact bugs found against real EMA documents.
"""

from __future__ import annotations

from medleaf.parse import (
    LEAFLET_HEADINGS,
    LEAFLET_WANTED,
    SMPC_HEADINGS,
    SMPC_WANTED,
    Line,
    _heading_at,
    _section_spans,
)


def _line(text: str, *, bold: bool, page: int = 1) -> Line:
    return Line(text=text, page=page, bold=bold, size=11.0)


def test_headings_are_found_from_font_weight_not_wording() -> None:
    """English and Greek headings look nothing alike, but both are bold."""
    lines = [
        _line("4.1", bold=True),
        _line("Therapeutic indications", bold=True),
        _line("Θεραπευτικές ενδείξεις", bold=False),
    ]
    assert _heading_at(lines, 0) == ("4.1", "Therapeutic indications", 2)


def test_number_only_heading_requires_a_bold_title() -> None:
    """A number followed by body text is not a heading."""
    lines = [
        _line("4.1", bold=True),
        _line("some ordinary body sentence", bold=False),
    ]
    assert _heading_at(lines, 0) is None


def test_table_row_number_is_not_a_heading_boundary() -> None:
    """'28 days' is a bold table header, not section 28 (real Eliquis 4.8).

    The low-level detector may see a number, but the heading vocabulary keeps
    it from cutting a section short.
    """
    lines = [
        _line("4.8", bold=True),
        _line("Undesirable effects", bold=True),
        _line("28 days", bold=True),
        _line("to less than 18 years of age", bold=True),
        _line("4.9", bold=True),
        _line("Overdose", bold=True),
    ]
    assert "28" not in SMPC_HEADINGS
    spans = _section_spans(lines, SMPC_WANTED, SMPC_HEADINGS)
    # 4.8 runs all the way to 4.9; the table header does not split it.
    assert spans["4.8"][2] == 4
    # 4.9 has no body in this stub, so it is correctly not recorded.
    assert set(spans) == {"4.8"}


def test_bullet_number_is_not_a_heading() -> None:
    """'3. Complete the following sections' is a leaflet bullet."""
    lines = [_line("3", bold=False), _line("Complete the following sections", bold=False)]
    assert _heading_at(lines, 0) is None


def test_page_number_is_not_a_heading() -> None:
    lines = [_line("143", bold=True), _line("Some bold line", bold=True)]
    assert _heading_at(lines, 0) is None


def test_smpc_section_boundaries_use_smpc_vocabulary() -> None:
    """6.4 must end at 6.5; a stray bold number must not cut 4.8 short."""
    lines = [
        _line("4.8", bold=True),
        _line("Undesirable effects", bold=True),
        _line("Nausea", bold=False),
        _line("28 days", bold=True),
        _line("more table text", bold=False),
        _line("4.9", bold=True),
        _line("Overdose", bold=True),
        _line("text", bold=False),
        _line("6.4", bold=True),
        _line("Special precautions for storage", bold=True),
        _line("Store below 25 C", bold=False),
        _line("6.5", bold=True),
        _line("Nature and contents of container", bold=True),
    ]
    spans = _section_spans(lines, SMPC_WANTED, SMPC_HEADINGS)
    assert spans["4.8"][0] == "Undesirable effects"
    # 4.8 body runs to the 4.9 heading, absorbing the "28 days" table line.
    assert spans["4.8"][2] == 5
    assert spans["6.4"][2] == 11


def test_leaflet_spans_ignore_smpc_numbering() -> None:
    """Leaflet 1-6 live in their own vocabulary, away from the 4.x numbering."""
    lines = [
        _line("1", bold=True),
        _line("What X is and what it is used for", bold=True),
        _line("body", bold=False),
        _line("2", bold=True),
        _line("What you need to know", bold=True),
        _line("body", bold=False),
    ]
    spans = _section_spans(lines, LEAFLET_WANTED, LEAFLET_HEADINGS)
    assert spans["1"][0] == "What X is and what it is used for"
    # Section 1's body ends where the 2 heading starts, at index 3.
    assert spans["1"][1] == 2
    assert spans["1"][2] == 3
    assert set(spans) == {"1", "2"}
