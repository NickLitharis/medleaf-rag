"""M2 acceptance: the real index, the real model and ten sample questions.

Opt in with both variables, because this needs a built index and the downloaded model:

    MEDLEAF_M2_CORPUS=1 MEDLEAF_QDRANT_URL=http://127.0.0.1:6333 \\
        PYTHONPATH=src python -m pytest tests/test_m2_corpus_index.py -q

Nothing here is skipped green: opting in without an index is a failure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from medleaf.config import Settings
from medleaf.ingest.store import load_medicines
from medleaf.models import Section
from medleaf.retrieval.embed import E5Encoder
from medleaf.retrieval.index import IndexMissingError, build_filter, corpus_version_for
from medleaf.retrieval.pipeline import ParsedCache, window_report
from medleaf.retrieval.service import RetrievalService

pytestmark = pytest.mark.qdrant

SMOKE_QUERIES = Path(__file__).parent / "data" / "m2_smoke_queries.json"
CROSS_LINGUAL_FLOOR = 0.70  # e5 cosine for a real cross-lingual match; hashing does not exceed ~0.6


def _settings() -> Settings:
    if os.environ.get("MEDLEAF_M2_CORPUS") != "1":
        pytest.skip("set MEDLEAF_M2_CORPUS=1 to run the real-index acceptance checks")
    url = os.environ.get("MEDLEAF_QDRANT_URL")
    if not url:
        pytest.skip("set MEDLEAF_QDRANT_URL to the built index")
    settings = Settings()
    return settings.model_copy(update={"qdrant_url": url})


def _service() -> RetrievalService:
    return RetrievalService(_settings())


def _smoke_queries() -> list[dict[str, object]]:
    payload = json.loads(SMOKE_QUERIES.read_text(encoding="utf-8"))
    return list(payload["queries"])


def test_manifest_records_the_pinned_real_model_and_the_whole_corpus() -> None:
    settings = _settings()
    manifest = _service().manifest()
    assert manifest.embed_model == settings.embed_model
    assert manifest.embed_revision == settings.embed_revision
    assert manifest.dense_dim == 768, "multilingual-e5-base is a 768-dimension model"
    assert manifest.sparse_modifier == "idf"
    assert manifest.missing == [], f"index build reported missing sources: {manifest.missing}"
    assert len(manifest.medicines) == 40, f"expected 40 medicines, got {len(manifest.medicines)}"
    assert manifest.total_chunks > 0
    assert manifest.corpus_version == corpus_version_for(settings.medicines_file)
    medicines = load_medicines(settings.medicines_file)
    assert len(medicines) == len(manifest.medicines)


@pytest.mark.parametrize("query", _smoke_queries(), ids=lambda q: str(q["id"]))
def test_sample_question_returns_its_section_in_the_top_five(
    query: dict[str, object],
) -> None:
    service = _service()
    response = service.search(
        str(query["question"]),
        lang=str(query["lang"]),
        medicine=[str(query["medicine"])],
        doc=str(query["doc"]),
        limit=5,
    )
    assert response.count > 0, "a real index must return hits for a corpus question"
    assert {hit.lang for hit in response.results} == {query["lang"]}
    assert all(hit.source_url.startswith("https://www.ema.europa.eu/") for hit in response.results)
    assert all(hit.text.strip() for hit in response.results)
    expected_raw = query["expected_sections"]
    assert isinstance(expected_raw, list)
    expected = {str(section) for section in expected_raw}
    sections = [hit.section for hit in response.results]
    assert expected & set(sections), (
        f"{query['id']}: expected one of {sorted(expected)} in the top 5, got {sections}"
    )


def test_language_filter_is_exact_on_the_real_index() -> None:
    """Real-index negative control: the lang filter is exact in *both* directions.

    Asserting only ``lang=el`` would pass even if the filter were ignored whenever the
    Greek query happened to rank Greek chunks first, so ``lang=en`` is checked too.
    """
    service = _service()
    for lang, other in (("el", "en"), ("en", "el")):
        response = service.search("ανεπιθύμητες ενέργειες ναυτία", lang=lang, limit=5)
        assert response.count > 0
        assert {hit.lang for hit in response.results} == {lang}, (
            f"lang={lang} returned {other} chunks too: "
            f"{[(hit.chunk_id, hit.lang) for hit in response.results]}"
        )
    unfiltered = service.search("ανεπιθύμητες ενέργειες ναυτία", limit=10)
    assert {hit.lang for hit in unfiltered.results} <= {"el", "en"}


def test_index_filters_reject_an_unknown_corpus_version() -> None:
    """An unindexed corpus version is an error, not a silently empty success."""
    service = _service()
    with pytest.raises(IndexMissingError):
        service.search("nausea", corpus_version="medicines.yaml:000000000000", limit=5)


def test_indexed_chunks_all_fit_the_model_window() -> None:
    """Every indexed chunk must be readable by the model: no silently truncated tails."""
    settings = _settings()
    service = _service()
    encoder = _load_encoder(settings)
    report = window_report(
        settings=settings, client=service.client, encoder=encoder, worst_limit=5
    )
    assert report.total_points > 0
    assert report.over_window == 0, (
        f"{report.over_window} chunks exceed {report.limit_tokens} tokens: "
        f"{report.by_medicine}; worst={report.worst}"
    )


def test_long_sections_keep_their_whole_text_in_the_index() -> None:
    """The corpus' longest sections must be indexed in full, not just their prefix."""
    settings = _settings()
    service = _service()
    checked = 0
    for slug, section_number in (("humira", "5.2"), ("jardiance", "4.8")):
        parts = _indexed_sections(service, settings, slug, "en", section_number)
        assert parts, f"{slug} en {section_number}: nothing indexed"
        indexed_text = _collapse(" ".join(str(payload["text"]) for payload in parts))
        parsed = _parsed_sections(settings, slug, "en")
        source_text = _collapse(
            " ".join(section.text for section in parsed if section.section == section_number)
        )
        assert source_text, f"{slug} en {section_number}: nothing parsed"
        assert indexed_text == source_text, (
            f"{slug} en {section_number}: the index covers {len(indexed_text)} characters "
            f"but the parse has {len(source_text)}"
        )
        totals = {int(payload["parts_total"]) for payload in parts}
        assert totals == {len(parts)}
        assert sorted(int(payload["part"]) for payload in parts) == list(range(1, len(parts) + 1))
        checked += 1
    assert checked == 2


def test_manifest_records_split_sections_metadata_and_versions() -> None:
    """The manifest must state the granularity split and the SPEC section 8 metadata."""
    settings = _settings()
    manifest = _service().manifest()
    assert manifest.split_sections, "long sections must be recorded, not hidden"
    assert all(parts >= 2 for parts in manifest.split_sections.values())
    assert manifest.active_substances, "manifest must carry the substance per medicine"
    assert manifest.atc_codes, "manifest must carry the ATC code per medicine"
    assert set(manifest.active_substances) == set(manifest.medicines)
    assert manifest.corpus_version == corpus_version_for(settings.medicines_file)


def _load_encoder(settings: Settings) -> E5Encoder:
    return E5Encoder.load(
        model_id=settings.embed_model,
        revision=settings.embed_revision,
        device=settings.embed_device,
        batch_size=settings.embed_batch_size,
        max_seq_length=settings.embed_max_seq_length,
        cache_dir=settings.embed_cache_dir,
    )


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _parsed_sections(settings: Settings, slug: str, lang: str) -> list[Section]:
    """Read the parse cache the index build wrote, so the comparison is like-for-like."""
    path = settings.parsed_path(slug, lang)
    try:
        cache = ParsedCache.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as error:
        pytest.fail(
            f"parse cache {path} predates the current parser cache version ({error}); "
            "this is the rebuild gate: the served index was built from the old parse. "
            "Re-run `medleaf index build --no-recreate --only <slugs>` and re-measure "
            "with `medleaf index window-report --json`."
        )
    return cache.sections


def _indexed_sections(
    service: RetrievalService, settings: Settings, slug: str, lang: str, number: str
) -> list[dict[str, object]]:
    points, _ = service.client.scroll(
        collection_name=settings.qdrant_collection,
        limit=256,
        with_payload=True,
        scroll_filter=build_filter(medicine=[slug], lang=lang, doc="smpc"),
    )
    payloads = [dict(point.payload or {}) for point in points]
    return sorted(
        (payload for payload in payloads if str(payload.get("section")) == number),
        key=lambda payload: int(payload.get("part", 1)),
    )


def test_real_model_is_cross_lingual_not_a_hash() -> None:
    """A real multilingual embedding puts a Greek question near its English passage."""
    settings = _settings()
    encoder = E5Encoder.load(
        model_id=settings.embed_model,
        revision=settings.embed_revision,
        device=settings.embed_device,
        batch_size=1,
        max_seq_length=settings.embed_max_seq_length,
        cache_dir=settings.embed_cache_dir,
    )
    query = encoder.encode_queries(["Πόσο συχνή είναι η ναυτία;"])[0]
    relevant, unrelated = encoder.encode_passages(
        [
            "Nausea is a very common adverse reaction.",
            "The marketing authorisation holder is Elli Lilly Nederland B.V.",
        ]
    )

    def cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True))

    matched = cosine(query, relevant)
    background = cosine(query, unrelated)
    assert matched > CROSS_LINGUAL_FLOOR, f"cross-lingual similarity too low: {matched:.3f}"
    assert matched > background, (
        f"Greek query must sit closer to the English passage ({matched:.3f}) than to an "
        f"unrelated one ({background:.3f})"
    )
