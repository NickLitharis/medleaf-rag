"""Paragraph rebuilding must not change the characters of clinical terms."""

from __future__ import annotations

import pytest

from medleaf.ingest.pdf import rebuild_paragraphs


@pytest.mark.parametrize(
    ("wrapped", "expected"),
    [
        ("Co-administration with P-\ngp inhibitors", "Co-administration with P-gp inhibitors"),
        ("HIV-\n1 infection", "HIV-1 infection"),
        ("5-\nHT receptor agonists", "5-HT receptor agonists"),
        ("non-\nsteroidal drugs", "non-steroidal drugs"),
    ],
)
def test_a_hyphen_at_a_line_end_is_kept(wrapped: str, expected: str) -> None:
    assert rebuild_paragraphs(wrapped) == expected


def test_wrapped_lines_join_into_paragraphs_separated_by_blank_lines() -> None:
    text = "Take one tablet\ndaily.\n\nDo not\ncrush it."
    assert rebuild_paragraphs(text) == "Take one tablet daily.\n\nDo not crush it."
