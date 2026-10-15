"""Shared value objects for the MedLeaf ingestion pipeline.

Everything in the pipeline is a plain, typed function over these objects so that
retrieval and evaluation can be exercised without the LLM (SPEC section 9).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl, model_validator


class Lang(StrEnum):
    """Corpus language. Greek is 'el', English is 'en'."""

    EL = "el"
    EN = "en"


class DocKind(StrEnum):
    """The two patient/professional documents parsed from one product-information PDF."""

    SMPC = "smpc"
    LEAFLET = "leaflet"


class DocumentRef(BaseModel):
    """One language version of one medicine's product-information PDF."""

    lang: Lang
    url: HttpUrl
    sha256: str | None = Field(
        default=None,
        description="SHA-256 of the downloaded PDF. None means 'not pinned'.",
    )
    revision_date: str | None = Field(
        default=None,
        description="Document revision date as printed by EMA (ISO 8601 when known).",
    )


class Medicine(BaseModel):
    """A centrally authorised medicine in the corpus."""

    slug: str = Field(description="Stable lowercase identifier, e.g. 'eliquis'.")
    name: str = Field(description="EMA brand name, e.g. 'Eliquis'.")
    active_substance: str
    atc_code: str | None = None
    epar_url: HttpUrl = Field(description="EMA EPAR landing page for the medicine.")
    name_el: str | None = Field(
        default=None,
        description="Brand name as written in Greek documents, for name matching.",
    )
    active_substance_el: str | None = Field(
        default=None,
        description="Active substance as written in Greek documents, for name matching.",
    )
    documents: dict[Lang, DocumentRef]

    @model_validator(mode="after")
    def _both_languages_present(self) -> Medicine:
        missing = {Lang.EL, Lang.EN} - set(self.documents)
        if missing:
            names = ", ".join(sorted(lang.value for lang in missing))
            raise ValueError(f"medicine {self.slug!r} is missing documents for: {names}")
        return self


class Section(BaseModel):
    """One numbered section of the SmPC or the package leaflet.

    The section ID is language-independent (SPEC section 8): the Greek and English
    versions of the same section share it, e.g. ``eliquis:smpc:4.5``.

    ``active_substance`` and ``atc_code`` are manifest metadata, not document text, so
    they are attached to the retrieval payload from the corpus manifest rather than
    stored on the section; a chunk can then never carry an invented substance name.
    """

    chunk_id: str
    medicine: str
    lang: Lang
    doc: DocKind
    section: str = Field(description="Numbered key without the medicine prefix, e.g. '4.5'.")
    title: str
    text: str
    source_url: str
    revision_date: str | None = None
    corpus_version: str | None = None
    part: int = Field(
        default=1,
        ge=1,
        description=(
            "1-based part index when a section is longer than one embedding window. "
            "Part 1 of 1 means the section was not split."
        ),
    )
    parts_total: int = Field(default=1, ge=1)

    @property
    def token_estimate(self) -> int:
        """Cheap length proxy used by the chunker's split threshold."""
        return len(self.text.split())


class ParsedDocument(BaseModel):
    """All sections found in one product-information PDF, in document order."""

    medicine: str
    lang: Lang
    source_path: Path
    source_url: str
    sections: list[Section]

    def by_doc(self, doc: DocKind) -> list[Section]:
        return [section for section in self.sections if section.doc is doc]
