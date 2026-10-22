"""CLI exit-code contracts for ``medleaf check``.

``check`` must fail loudly when the corpus manifest is missing or when no
medicine/language rows are selected; a misleading "0/0 pairs complete" exit 0 is not
an acceptable result for a verification command.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from medleaf import cli
from medleaf.config import Settings


def test_check_fails_when_no_medicines_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(data_dir=tmp_path))

    assert cli.main(["check"]) == 1
    captured = capsys.readouterr()
    assert "no medicines" in captured.err
    assert "0/0 pairs complete" not in captured.out


def test_check_fails_when_no_rows_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(data_dir=tmp_path))
    monkeypatch.setattr(
        cli, "load_medicines", lambda _path: [SimpleNamespace(slug="eliquis")]
    )

    assert cli.main(["check", "--only", "not-a-medicine"]) == 1
    captured = capsys.readouterr()
    assert "no medicine-language rows" in captured.err
    assert "0/0 pairs complete" not in captured.out
