"""Fast unit tests for embedding prefixes, chunk splitting and the parse cache.

These need no torch and no network: they pin the e5 prefix rule, how long sections are
split to fit the model window, and when a cached parse may be reused.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pytest

from medleaf.config import Settings
from medleaf.embed import PASSAGE_PREFIX, QUERY_PREFIX, E5Encoder, prefixed
from medleaf.ingest.cache import (
    CorpusError,
    ParsedCache,
    cache_is_valid,
    parse_with_cache,
    resolved_corpus_version,
)
from medleaf.ingest.chunk import ChunkBudget, build_sections, split_long_text
from medleaf.ingest.pipeline import PARSER_CACHE_VERSION
from medleaf.ingest.store import load_medicines
from medleaf.models import DocKind, DocumentRef, Lang, Medicine, Section


def _collapse(text: str) -> str:
    """Whitespace-insensitive text, so a split/join round trip can be compared."""
    return re.sub(r"\s+", " ", text).strip()


def _word_budget(max_tokens: int) -> ChunkBudget:
    return ChunkBudget(
        max_tokens=max_tokens,
        reserve_tokens=0,
        count_tokens=lambda text: len(text.split()),
        label="test-words",
    )

#: The M2 corpus target. The real count is asserted against data/medicines.yaml.
EXPECTED_MEDICINES = 40


class _RecordingModel:
    """Stands in for a sentence-transformers model and records prefixed inputs."""

    def __init__(self, dimension: int = 4) -> None:
        self.inputs: list[list[str]] = []
        self._dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        self.inputs.append(list(texts))
        return [[0.0] * self._dimension for _ in texts]
















def test_prefixed_rejects_a_query_passed_as_a_passage() -> None:
    """Failure-sensitive: the mismatched prefix must raise, not silently degrade."""
    with pytest.raises(ValueError, match="query:"):
        prefixed([f"{QUERY_PREFIX}how common is nausea"], PASSAGE_PREFIX, forbidden=QUERY_PREFIX)
    assert prefixed(["how common is nausea"], PASSAGE_PREFIX, forbidden=QUERY_PREFIX) == [
        "passage: how common is nausea"
    ]


def test_e5_encoder_applies_the_right_prefix_per_call() -> None:
    model = _RecordingModel()
    encoder = E5Encoder(model, model_id="intfloat/multilingual-e5-base", revision="abc123")
    assert encoder.encode_passages(["Nausea is very common"])
    assert encoder.encode_queries(["How common is nausea?", "Πόσο συχνή είναι η ναυτία;"])
    assert model.inputs[0] == [f"{PASSAGE_PREFIX}Nausea is very common"]
    assert model.inputs[1] == [
        f"{QUERY_PREFIX}How common is nausea?",
        f"{QUERY_PREFIX}Πόσο συχνή είναι η ναυτία;",
    ]
    assert encoder.dimension == 4


def test_e5_encoder_refuses_a_query_string_in_the_passage_path() -> None:
    encoder = E5Encoder(_RecordingModel(), model_id="m", revision="r")
    with pytest.raises(ValueError, match="query:"):
        encoder.encode_passages([f"{QUERY_PREFIX}leaked"])
    with pytest.raises(ValueError, match="passage:"):
        encoder.encode_queries([f"{PASSAGE_PREFIX}leaked"])












def test_m2_corpus_manifest_has_the_target_medicines_both_languages_and_pins() -> None:
    settings = Settings()
    medicines = load_medicines(settings.medicines_file)
    assert len(medicines) >= EXPECTED_MEDICINES, (
        f"M2 target is {EXPECTED_MEDICINES} medicines, found {len(medicines)}"
    )
    assert len({medicine.slug for medicine in medicines}) == len(medicines)
    slugs = {medicine.slug for medicine in medicines}
    assert {"eliquis", "humira", "jardiance", "ozempic", "xarelto"} <= slugs, "M1 kept"
    for medicine in medicines:
        assert set(medicine.documents) == {"en", "el"}, f"{medicine.slug} needs both languages"
        for lang, document in medicine.documents.items():
            assert document.sha256, f"{medicine.slug}/{lang} must be pinned"
            assert document.revision_date, f"{medicine.slug}/{lang} must carry a revision date"
            assert document.url.host == "www.ema.europa.eu"
    therapy_areas = {medicine.atc_code[:1] for medicine in medicines if medicine.atc_code}
    assert len(therapy_areas) >= 8, f"expected several therapy areas, got {sorted(therapy_areas)}"




# --- Long sections must be split, never silently truncated -------------------------


def test_split_long_text_respects_the_budget_and_keeps_every_word() -> None:
    budget = _word_budget(20)
    text = "\n\n".join(f"Paragraph {index} " + "word " * 18 for index in range(3))
    parts = split_long_text(text, budget)
    assert len(parts) >= 3
    assert all(budget.fits(part) for part in parts)
    assert _collapse("\n".join(parts)) == _collapse(text)


def test_oversized_single_paragraph_is_split_not_truncated() -> None:
    """The old splitter kept a huge paragraph whole; the model then truncated it."""
    budget = _word_budget(20)
    text = "word " * 60
    parts = split_long_text(text, budget)
    assert len(parts) >= 3, f"a 60-word paragraph must not become one chunk: {len(parts)}"
    assert all(budget.fits(part) for part in parts)
    assert _collapse("\n".join(parts)) == _collapse(text)


def test_table_rows_stay_intact_when_a_section_is_split() -> None:
    budget = _word_budget(20)
    rows = ["Nausea - very common", "Vomiting - common", "Headache - uncommon"]
    text = "\n\n".join([*rows, "paragraph " + "word " * 40])
    parts = split_long_text(text, budget)
    assert len(parts) > 1
    for row in rows:
        assert any(row in part for part in parts), f"row split across parts: {row}"
    assert _collapse("\n".join(parts)) == _collapse(text)


def test_build_sections_number_the_parts_and_keep_the_base_section() -> None:
    text = "\n\n".join("word " * 30 for _ in range(3))
    sections = build_sections(
        medicine="demo",
        lang=Lang.EN,
        doc=DocKind.SMPC,
        raw_sections=[("4.8", "Undesirable effects", text)],
        source_url="https://example.invalid/x.pdf",
        revision_date="2026-01-01",
        corpus_version="test",
        max_tokens=25,
    )
    assert len(sections) > 1
    assert {section.section for section in sections} == {"4.8"}
    assert [section.chunk_id for section in sections] == [
        f"demo:smpc:4.8#{index}" for index in range(1, len(sections) + 1)
    ]
    assert [section.part for section in sections] == list(range(1, len(sections) + 1))
    assert {section.parts_total for section in sections} == {len(sections)}


def test_unsplit_section_keeps_its_bare_id_and_part_one() -> None:
    sections = build_sections(
        medicine="demo",
        lang=Lang.EL,
        doc=DocKind.LEAFLET,
        raw_sections=[("3", "Πώς να πάρετε", "Πάρτε το χάπι.")],
        source_url="https://example.invalid/x.pdf",
        revision_date=None,
        corpus_version="test",
        max_tokens=100,
    )
    assert len(sections) == 1
    assert sections[0].chunk_id == "demo:leaflet:3"
    assert sections[0].part == 1 and sections[0].parts_total == 1


# --- Parse cache invalidation and corpus-version stamping --------------------------


def _cached(
    *,
    cache_version: str = PARSER_CACHE_VERSION,
    pdf_sha256: str = "a" * 64,
    budget_label: str = "budget-1",
) -> ParsedCache:
    return ParsedCache(
        cache_version=cache_version,
        pdf_sha256=pdf_sha256,
        budget_label=budget_label,
        corpus_version="v",
        source_url="https://example.invalid/x.pdf",
        sections=[],
    )


def test_cache_is_valid_only_when_pdf_parser_and_budget_all_match() -> None:
    assert cache_is_valid(_cached(), pdf_sha256="a" * 64, budget_label="budget-1")
    # Negatives: each is a deliberate change that must invalidate the cache.
    assert not cache_is_valid(None, pdf_sha256="a" * 64, budget_label="budget-1")
    assert not cache_is_valid(
        _cached(cache_version="medleaf-parser-1"),
        pdf_sha256="a" * 64,
        budget_label="budget-1",
    ), "a parser/chunker fix must invalidate old caches"
    assert not cache_is_valid(
        _cached(pdf_sha256="b" * 64), pdf_sha256="a" * 64, budget_label="budget-1"
    ), "a changed PDF must invalidate the cache"
    assert not cache_is_valid(
        _cached(budget_label="other"), pdf_sha256="a" * 64, budget_label="budget-1"
    ), "a different embedding window must invalidate the cache"
    assert PARSER_CACHE_VERSION, "the parser cache version must be set"


def _pinned_medicine(sha256: str) -> Medicine:
    return Medicine(
        slug="demo",
        name="Demo",
        active_substance="demo",
        epar_url="https://www.ema.europa.eu/en/medicines/human/EPAR/demo",
        documents={
            lang: DocumentRef(
                lang=lang,
                url=(
                    "https://www.ema.europa.eu/"
                    f"{lang.value}/documents/product-information/"
                    f"demo-epar-product-information_{lang.value}.pdf"
                ),
                sha256=sha256,
            )
            for lang in (Lang.EN, Lang.EL)
        },
    )


def test_pinned_checksum_mismatch_aborts_before_parsing(tmp_path: Path) -> None:
    """A changed PDF must not be indexed under the old corpus pin."""
    settings = Settings(data_dir=tmp_path, corpus_version="test")
    pdf = settings.pdf_path("demo", "en")
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"not the pinned PDF")
    medicine = _pinned_medicine("a" * 64)

    with pytest.raises(CorpusError, match="is pinned to"):
        parse_with_cache(settings, medicine, Lang.EN, budget=_word_budget(512))
    assert not settings.parsed_path("demo", "en").exists(), "no cache for a rejected PDF"


def test_resolved_corpus_version_prefers_an_explicit_setting(tmp_path: Path) -> None:
    (tmp_path / "medicines.yaml").write_text("medicines: []\n", encoding="utf-8")
    derived = resolved_corpus_version(Settings(data_dir=tmp_path, corpus_version="dev"))
    assert derived.startswith("medicines.yaml:")
    explicit = resolved_corpus_version(Settings(data_dir=tmp_path, corpus_version="corpus-7"))
    assert explicit == "corpus-7"


def test_parse_cache_restamps_the_corpus_version_on_a_switch(tmp_path: Path) -> None:
    """Same PDF, two corpus versions: the cache is reused and re-stamped, never stale."""
    real = Settings()
    pdf = real.pdf_path("eliquis", "el")
    if not pdf.exists():
        if os.environ.get("MEDLEAF_M1_CORPUS") == "1":
            pytest.fail(f"MEDLEAF_M1_CORPUS=1 but {pdf} is missing")
        pytest.skip("corpus PDF not downloaded; set MEDLEAF_M1_CORPUS=1 to require it")
    medicine = next(m for m in load_medicines(real.medicines_file) if m.slug == "eliquis")

    settings_a = Settings(data_dir=tmp_path, corpus_version="corpus-a")
    target = settings_a.pdf_path("eliquis", "el")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, target)
    budget = ChunkBudget(
        max_tokens=512,
        reserve_tokens=16,
        count_tokens=lambda text: max(1, len(text) // 4),
        label="test-budget",
    )
    first = parse_with_cache(settings_a, medicine, Lang.EL, budget=budget)
    assert first.origin == "parsed"
    assert first.sections
    assert {section.corpus_version for section in first.sections} == {"corpus-a"}

    settings_b = settings_a.model_copy(update={"corpus_version": "corpus-b"})
    second = parse_with_cache(settings_b, medicine, Lang.EL, budget=budget)
    assert second.origin == "cache", "the unchanged PDF must not be re-parsed"
    assert {section.corpus_version for section in second.sections} == {"corpus-b"}, (
        "cached sections must be stamped with the requested version, not the cached one"
    )
    assert [section.chunk_id for section in second.sections] == [
        section.chunk_id for section in first.sections
    ]


# --- Chunk payload metadata (SPEC section 8) ---------------------------------------


def _payload_section(**overrides: object) -> Section:
    values: dict[str, object] = {
        "chunk_id": "eliquis:smpc:4.8#2",
        "medicine": "eliquis",
        "lang": Lang.EN,
        "doc": DocKind.SMPC,
        "section": "4.8",
        "title": "Undesirable effects (part 2)",
        "text": "Nausea - very common",
        "source_url": "https://www.ema.europa.eu/en/documents/product-information/eliquis-epar-product-information_en.pdf",
        "revision_date": "2026-04-01",
        "corpus_version": "v1",
        "part": 2,
        "parts_total": 3,
    }
    values.update(overrides)
    return Section.model_validate(values)




