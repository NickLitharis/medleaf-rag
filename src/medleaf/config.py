"""Configuration: environment variables only (SPEC section 7)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings.

    Values come from the environment or a local ``.env`` file. Field names map to
    upper-case environment variables, e.g. ``MEDLEAF_DATA_DIR``.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEDLEAF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    configs_dir: Path = Field(default=PROJECT_ROOT / "configs")
    corpus_version: str = Field(default="dev")
    user_agent: str = Field(
        default="medleaf-rag/0.1 (+https://github.com/NickLitharis/medleaf-rag)"
    )
    request_timeout_s: float = Field(default=60.0)
    max_section_tokens: int = Field(default=400)

    def pdf_path(self, slug: str, lang: str) -> Path:
        """Path of one downloaded PDF, matching ``ingest.download.download_medicine``."""
        return self.raw_dir / slug / f"{slug}_{lang}.pdf"

    @property
    def raw_dir(self) -> Path:
        """Directory holding downloaded PDFs (gitignored)."""
        return self.data_dir / "raw"

    @property
    def parsed_dir(self) -> Path:
        """Directory holding parsed chunk JSONL."""
        return self.data_dir / "parsed"

    @property
    def medicines_file(self) -> Path:
        """Committed corpus manifest."""
        return self.data_dir / "medicines.yaml"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
