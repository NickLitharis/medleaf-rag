"""Corpus manifest models and validation (SPEC section 5 and 8)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

Lang = Literal["en", "el"]
Doc = Literal["smpc", "leaflet"]

SUPPORTED_LANGS: tuple[Lang, ...] = ("en", "el")

# EMA product-information PDFs live under a language-prefixed path. Using the
# wrong prefix is not a 404: /en/...<name>_el.pdf answers 301 and serves the
# ENGLISH document. Always build the URL from the document language.
EMA_PI_URL = "https://www.ema.europa.eu/{lang}/documents/product-information/{slug}_epar-product-information_{lang}.pdf"


class SourceFile(BaseModel):
    """One downloaded document: URL plus the checksum of the bytes we parsed."""

    model_config = ConfigDict(extra="forbid")

    url: str
    sha256: str
    revision_date: str
    bytes: int = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def _hex_digest(cls, value: str) -> str:
        normalised = value.strip().lower()
        if len(normalised) != 64 or any(c not in "0123456789abcdef" for c in normalised):
            msg = "sha256 must be a 64-character lower-case hex digest"
            raise ValueError(msg)
        return normalised

    def short_sha256(self) -> str:
        """First 12 hex characters, used in logs and filenames."""
        return self.sha256[:12]


class Medicine(BaseModel):
    """A centrally authorised medicine and its official EMA page."""

    model_config = ConfigDict(extra="forbid")

    slug: str
    name: str
    active_substance: str
    atc_code: str
    ema_url: str
    sources: dict[Lang, SourceFile] = Field(default_factory=dict)

    @property
    def chunk_prefix(self) -> str:
        """Chunk-ID prefix; stable and language-independent."""
        return self.slug.lower()

    def source(self, lang: Lang) -> SourceFile:
        """Return the source record for a language, failing loudly when absent."""
        try:
            return self.sources[lang]
        except KeyError:
            msg = f"{self.slug}: no {lang} source recorded; run `medleaf fetch` first"
            raise LookupError(msg) from None

    def has_both_languages(self) -> bool:
        """True when English and Greek sources are both present."""
        return all(lang in self.sources for lang in SUPPORTED_LANGS)


class Corpus(BaseModel):
    """The committed medicine manifest."""

    model_config = ConfigDict(extra="forbid")

    corpus_version: str
    medicines: list[Medicine]

    def by_slug(self, slug: str) -> Medicine:
        """Look a medicine up by its slug."""
        lowered = slug.lower()
        for medicine in self.medicines:
            if medicine.slug.lower() == lowered:
                return medicine
        msg = f"unknown medicine slug: {slug}"
        raise KeyError(msg)

    def slugs(self) -> list[str]:
        """All medicine slugs in manifest order."""
        return [m.slug for m in self.medicines]


def product_information_url(slug: str, lang: Lang) -> str:
    """Build the correct EMA product-information URL for a language.

    The path language and the filename suffix must both match ``lang``. The
    suffix alone is not enough: a mismatched path redirects to English.
    """
    return EMA_PI_URL.format(lang=lang, slug=slug)


def load_corpus(path: Path) -> Corpus:
    """Read and validate the manifest at ``path``."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"{path}: expected a YAML mapping at the top level"
        raise ValueError(msg)
    return Corpus.model_validate(raw)


def dump_corpus(corpus: Corpus, path: Path) -> None:
    """Write the manifest back, preserving a readable field order."""
    payload = corpus.model_dump(mode="json", exclude_defaults=False)
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=100)
    path.write_text(text, encoding="utf-8")
