"""Hybrid-retrieval contract tests against Qdrant.

These run against ``MEDLEAF_QDRANT_URL`` when set, and otherwise against Qdrant's
in-memory mode; if that build cannot do sparse IDF plus RRF the tests skip rather than
pretend. The embedding model here is a hash-based *test double* — it exists to make
dense scores deterministic, and the real multilingual model is exercised by
``tests/test_m2_corpus_index.py`` under its own opt-in gate.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from medleaf.config import Settings
from medleaf.ingest.chunk import ChunkBudget
from medleaf.models import DocKind, Lang, Medicine, Section
from medleaf.retrieval import pipeline
from medleaf.retrieval.embed import E5Encoder, PrefixEnforcingEncoder
from medleaf.retrieval.index import (
    IndexIncompleteError,
    IndexMissingError,
    RetrievalUnavailableError,
    alias_target,
    build_filter,
    build_manifest,
    collection_count,
    delete_medicine_points,
    ensure_collection,
    hybrid_search,
    load_manifest,
    point_alias,
    point_id,
    require_index,
    set_medicine_metadata,
    upsert_sections,
    write_manifest,
)
from medleaf.retrieval.pipeline import IndexBuildError, ParseResult, build_index
from medleaf.retrieval.service import RetrievalService
from medleaf.retrieval.text import sparse_vector, token_id, tokenize

pytestmark = pytest.mark.qdrant

DIMENSION = 64
DENSE = "dense"
SPARSE = "sparse"


class HashTestEncoder:
    """Bag-of-words hashing test double with the real encoder's interface.

    Deterministic by construction (blake2b over folded tokens) and lexically sensible,
    so a chunk that literally contains the query text really does win the dense channel.
    This is a test double, not the production encoder: the real multilingual model is
    exercised by ``tests/test_m2_corpus_index.py``.
    """

    model_id = "test/hash-double"
    revision = "test"

    @property
    def dimension(self) -> int:
        return DIMENSION

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSION
        for token in tokenize(text):
            vector[token_id(token) % DIMENSION] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Crude four-characters-per-token count; only used for protocol conformance."""
        return [max(1, len(text) // 4) for text in texts]


def _section(
    *,
    medicine: str,
    lang: Lang,
    doc: DocKind,
    number: str,
    text: str,
    corpus_version: str = "v1",
) -> Section:
    return Section(
        chunk_id=f"{medicine}:{doc.value}:{number}",
        medicine=medicine,
        lang=lang,
        doc=doc,
        section=number,
        title=f"Section {number}",
        text=text,
        source_url=f"https://www.ema.europa.eu/{lang.value}/documents/product-information/{medicine}-epar-product-information_{lang.value}.pdf",
        revision_date="2026-01-01",
        corpus_version=corpus_version,
    )


def _local_mode_supported(client: QdrantClient) -> bool:
    """Probe whether this Qdrant build can run the production contract."""
    probe = f"medleaf_probe_{uuid.uuid4().hex[:8]}"
    try:
        ensure_collection(
            client,
            collection=probe,
            dense_dim=4,
            dense_vector_name=DENSE,
            sparse_vector_name=SPARSE,
            recreate=True,
        )
        upsert_sections(
            client,
            collection=probe,
            sections=[
                _section(
                    medicine="probe",
                    lang=Lang.EN,
                    doc=DocKind.SMPC,
                    number="4.3",
                    text="Contraindications include active bleeding.",
                )
            ],
            dense_vectors=[[1.0, 0.0, 0.0, 0.0]],
            dense_vector_name=DENSE,
            sparse_vector_name=SPARSE,
        )
        hybrid_search(
            client,
            collection=probe,
            dense_vector=[1.0, 0.0, 0.0, 0.0],
            sparse_query=sparse_vector("active bleeding"),
            dense_vector_name=DENSE,
            sparse_vector_name=SPARSE,
            limit=1,
        )
    finally:
        if client.collection_exists(probe):
            client.delete_collection(probe)
    return True


@pytest.fixture
def qdrant() -> Iterator[QdrantClient]:
    url = os.environ.get("MEDLEAF_QDRANT_URL")
    if url:
        client = QdrantClient(url=url, timeout=30)
        try:
            client.get_collections()
        except Exception as error:  # pragma: no cover - environment dependent
            pytest.skip(f"Qdrant not reachable at {url}: {error}")
        yield client
        client.close()
        return
    client = QdrantClient(":memory:")
    try:
        _local_mode_supported(client)
    except Exception as error:  # pragma: no cover - depends on the installed client
        client.close()
        pytest.skip(f"in-memory Qdrant cannot run the hybrid contract: {error}")
    yield client
    client.close()


@pytest.fixture
def collection(qdrant: QdrantClient) -> Iterator[str]:
    name = f"medleaf_test_{uuid.uuid4().hex[:10]}"
    yield name
    if qdrant.collection_exists(name):
        qdrant.delete_collection(name)


def _seed_eliquis(qdrant: QdrantClient, collection: str, encoder: HashTestEncoder) -> None:
    sections = [
        _section(
            medicine="eliquis",
            lang=Lang.EN,
            doc=DocKind.SMPC,
            number="4.3",
            text="Contraindications: active clinically significant bleeding.",
        ),
        _section(
            medicine="eliquis",
            lang=Lang.EL,
            doc=DocKind.SMPC,
            number="4.3",
            text="Αντενδείξεις: ενεργή κλινικά σημαντική αιμορραγία.",
        ),
        _section(
            medicine="eliquis",
            lang=Lang.EN,
            doc=DocKind.SMPC,
            number="4.8",
            text="Nausea is a common adverse reaction.",
        ),
        _section(
            medicine="xarelto",
            lang=Lang.EN,
            doc=DocKind.SMPC,
            number="4.3",
            text="Contraindications: active bleeding and hepatic disease.",
            corpus_version="v2",
        ),
    ]
    ensure_collection(
        qdrant,
        collection=collection,
        dense_dim=encoder.dimension,
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
        recreate=True,
    )
    upsert_sections(
        qdrant,
        collection=collection,
        sections=sections,
        dense_vectors=encoder.encode_passages([section.text for section in sections]),
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
    )


def test_collection_declares_sparse_idf_vectors(qdrant: QdrantClient, collection: str) -> None:
    ensure_collection(
        qdrant,
        collection=collection,
        dense_dim=DIMENSION,
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
    )
    info = qdrant.get_collection(collection)
    sparse_config = info.config.params.sparse_vectors or {}
    assert SPARSE in sparse_config
    assert sparse_config[SPARSE].modifier == models.Modifier.IDF
    dense_config = info.config.params.vectors
    assert isinstance(dense_config, dict)
    assert dense_config[DENSE].size == DIMENSION


def test_language_filter_is_exact_and_the_negative_control_proves_it(
    qdrant: QdrantClient, collection: str
) -> None:
    """Filtering by ``lang`` must exclude the other language — and the test can fail.

    The second half searches the same collection *without* the filter and asserts the
    Greek chunk does appear. If the filter were ignored, the first assertion would still
    pass, so this second assertion is what makes the check meaningful.
    """
    encoder = HashTestEncoder()
    _seed_eliquis(qdrant, collection, encoder)
    dense = encoder.encode_queries(["contraindications active bleeding"])[0]
    sparse = sparse_vector("contraindications active bleeding")

    english_only = hybrid_search(
        qdrant,
        collection=collection,
        dense_vector=dense,
        sparse_query=sparse,
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
        limit=10,
        query_filter=build_filter(lang="en"),
    )
    assert english_only, "the English filter must still return matches"
    assert {point.payload["lang"] for point in english_only if point.payload} == {"en"}

    unfiltered = hybrid_search(
        qdrant,
        collection=collection,
        dense_vector=dense,
        sparse_query=sparse,
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
        limit=10,
    )
    assert "el" in {point.payload["lang"] for point in unfiltered if point.payload}


def test_medicine_and_corpus_version_filters_are_exact(
    qdrant: QdrantClient, collection: str
) -> None:
    encoder = HashTestEncoder()
    _seed_eliquis(qdrant, collection, encoder)
    dense = encoder.encode_queries(["contraindications bleeding"])[0]
    sparse = sparse_vector("contraindications bleeding")

    def search(query_filter: models.Filter | None) -> list[dict[str, object]]:
        points = hybrid_search(
            qdrant,
            collection=collection,
            dense_vector=dense,
            sparse_query=sparse,
            dense_vector_name=DENSE,
            sparse_vector_name=SPARSE,
            limit=10,
            query_filter=query_filter,
        )
        return [dict(point.payload or {}) for point in points]

    by_medicine = search(build_filter(medicine=["xarelto"], lang="en"))
    assert by_medicine, "xarelto/v2 must be retrievable"
    assert {payload["medicine"] for payload in by_medicine} == {"xarelto"}

    by_version = search(build_filter(corpus_version="v1"))
    assert {payload["corpus_version"] for payload in by_version} == {"v1"}
    assert {payload["medicine"] for payload in by_version} == {"eliquis"}


def test_rrf_fuses_sparse_and_dense_candidates(qdrant: QdrantClient, collection: str) -> None:
    """A chunk matched by both channels outranks chunks matched by only one.

    Two searches are run: the first proves both channels contribute candidates (the
    sparse-only chunk is there only because the sparse channel found it), the second
    proves the chunk really was sparse-only, because an unrelated sparse query drops it.
    """
    encoder = HashTestEncoder()
    ensure_collection(
        qdrant,
        collection=collection,
        dense_dim=encoder.dimension,
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
        recreate=True,
    )
    dense_text = "Prevention of stroke is the therapeutic indication."
    both = _section(
        medicine="eliquis",
        lang=Lang.EN,
        doc=DocKind.SMPC,
        number="4.5",
        text=f"{dense_text} Υπερδοσολογία is discussed separately.",
    )
    dense_only = _section(
        medicine="eliquis", lang=Lang.EN, doc=DocKind.SMPC, number="4.1", text=dense_text
    )
    sparse_only = _section(
        medicine="eliquis",
        lang=Lang.EL,
        doc=DocKind.SMPC,
        number="4.9",
        text="Υπερδοσολογία: δεν υπάρχει ειδικό αντίδοτο.",
    )
    filler = [
        _section(
            medicine="eliquis",
            lang=Lang.EN,
            doc=DocKind.SMPC,
            number=f"6.{index}",
            text=f"Filler {index} covers excipients, shelf life and disposal of the container.",
        )
        for index in range(1, 7)
    ]
    sections = [both, dense_only, sparse_only, *filler]
    upsert_sections(
        qdrant,
        collection=collection,
        sections=sections,
        dense_vectors=encoder.encode_passages([section.text for section in sections]),
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
    )

    def search(sparse_text: str, limit: int) -> list[str]:
        points = hybrid_search(
            qdrant,
            collection=collection,
            dense_vector=encoder.encode_queries([dense_text])[0],
            sparse_query=sparse_vector(sparse_text),
            dense_vector_name=DENSE,
            sparse_vector_name=SPARSE,
            limit=limit,
            prefetch_limit=5,
        )
        return [str(point.payload["chunk_id"]) for point in points if point.payload]

    ids = search("ΥΠΕΡΔΟΣΟΛΟΓΊΑ", 10)
    assert dense_only.chunk_id in ids, "the dense channel must contribute its match"
    assert sparse_only.chunk_id in ids, "the sparse channel must contribute its match"
    assert ids[0] == both.chunk_id, (
        f"the chunk found by both channels must win RRF, got order {ids}"
    )

    unrelated = search("ασυμβατότητες", 5)
    assert sparse_only.chunk_id not in unrelated, (
        "with an unrelated sparse query the sparse-only chunk must drop out, otherwise "
        "the sparse channel is not doing the work"
    )


def test_sparse_tokens_fold_greek_accents_end_to_end(
    qdrant: QdrantClient, collection: str
) -> None:
    encoder = HashTestEncoder()
    target = _section(
        medicine="ozempic",
        lang=Lang.EL,
        doc=DocKind.SMPC,
        number="4.8",
        text="Ανεπιθύμητες ενέργειες: υπερδοσολογία αναφέρθηκε σπάνια.",
    )
    other = _section(
        medicine="ozempic",
        lang=Lang.EL,
        doc=DocKind.SMPC,
        number="4.1",
        text="Θεραπευτικές ενδείξεις για τον διαβήτη τύπου 2.",
    )
    ensure_collection(
        qdrant,
        collection=collection,
        dense_dim=encoder.dimension,
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
        recreate=True,
    )
    upsert_sections(
        qdrant,
        collection=collection,
        sections=[target, other],
        dense_vectors=encoder.encode_passages([target.text, other.text]),
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
    )
    # Uppercase, accented query must match the stored accented text via folded tokens.
    results = hybrid_search(
        qdrant,
        collection=collection,
        # Deliberately-unrelated dense query so the sparse channel decides the top hit.
        dense_vector=encoder.encode_queries(["zzz unrelated dense query"])[0],
        sparse_query=sparse_vector("ΑΝΕΠΙΘΎΜΗΤΕΣ ΕΝΈΡΓΕΙΕΣ"),
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
        limit=2,
    )
    assert results[0].payload["section"] == "4.8"  # type: ignore[index]


def test_ids_are_unique_per_language_and_upsert_is_idempotent(
    qdrant: QdrantClient, collection: str
) -> None:
    encoder = HashTestEncoder()
    english = _section(
        medicine="eliquis", lang=Lang.EN, doc=DocKind.SMPC, number="4.3", text="English text."
    )
    greek = _section(
        medicine="eliquis", lang=Lang.EL, doc=DocKind.SMPC, number="4.3", text="Ελληνικό κείμενο."
    )
    assert point_id(english.chunk_id, "en") != point_id(greek.chunk_id, "el")
    ensure_collection(
        qdrant,
        collection=collection,
        dense_dim=encoder.dimension,
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
        recreate=True,
    )
    for _ in range(2):
        upsert_sections(
            qdrant,
            collection=collection,
            sections=[english, greek],
            dense_vectors=encoder.encode_passages([english.text, greek.text]),
            dense_vector_name=DENSE,
            sparse_vector_name=SPARSE,
        )
    assert collection_count(qdrant, collection=collection) == 2
    assert (
        collection_count(qdrant, collection=collection, query_filter=build_filter(lang="el")) == 1
    )


def test_index_persists_across_a_new_client(qdrant: QdrantClient, collection: str) -> None:
    url = os.environ.get("MEDLEAF_QDRANT_URL")
    if not url:
        pytest.skip("persistence needs a real Qdrant service, not in-memory mode")
    encoder = HashTestEncoder()
    _seed_eliquis(qdrant, collection, encoder)
    reopened = QdrantClient(url=url, timeout=30)
    try:
        assert collection_count(reopened, collection=collection) == 4
        assert reopened.collection_exists(collection)
        assert reopened.get_collection(collection).points_count == 4
    finally:
        reopened.close()


def test_payload_carries_manifest_metadata_and_is_filterable(
    qdrant: QdrantClient, collection: str
) -> None:
    """SPEC section 8 metadata travels with the chunk and can be filtered on."""
    encoder = HashTestEncoder()
    sections = [
        _section(
            medicine="eliquis",
            lang=Lang.EN,
            doc=DocKind.SMPC,
            number="4.3",
            text="Contraindications: active clinically significant bleeding.",
        ),
        _section(
            medicine="xarelto",
            lang=Lang.EN,
            doc=DocKind.SMPC,
            number="4.3",
            text="Contraindications: active bleeding and hepatic disease.",
        ),
    ]
    ensure_collection(
        qdrant,
        collection=collection,
        dense_dim=encoder.dimension,
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
        recreate=True,
    )
    upsert_sections(
        qdrant,
        collection=collection,
        sections=sections,
        dense_vectors=encoder.encode_passages([section.text for section in sections]),
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
        metadata={
            "eliquis": {"active_substance": "apixaban", "atc_code": "B01AF02"},
            "xarelto": {"active_substance": "rivaroxaban", "atc_code": "B01AF01"},
        },
    )
    points, _ = qdrant.scroll(collection_name=collection, limit=10, with_payload=True)
    by_slug = {str(point.payload["medicine"]): point.payload for point in points if point.payload}
    assert by_slug["eliquis"]["active_substance"] == "apixaban"
    assert by_slug["eliquis"]["atc_code"] == "B01AF02"
    assert by_slug["eliquis"]["part"] == 1
    assert by_slug["eliquis"]["parts_total"] == 1

    dense = encoder.encode_queries(["contraindications bleeding"])[0]
    hits = hybrid_search(
        qdrant,
        collection=collection,
        dense_vector=dense,
        sparse_query=sparse_vector("contraindications bleeding"),
        dense_vector_name=DENSE,
        sparse_vector_name=SPARSE,
        limit=5,
        query_filter=build_filter(active_substance="apixaban", lang="en"),
    )
    assert {str(point.payload["medicine"]) for point in hits if point.payload} == {"eliquis"}
    # Negative control: a substance no chunk carries must match nothing, not fall back to
    # an unfiltered search.
    assert (
        hybrid_search(
            qdrant,
            collection=collection,
            dense_vector=dense,
            sparse_query=sparse_vector("contraindications bleeding"),
            dense_vector_name=DENSE,
            sparse_vector_name=SPARSE,
            limit=5,
            query_filter=build_filter(active_substance="semaglutide"),
        )
        == []
    )


def test_set_medicine_metadata_updates_payload_without_reembedding(
    qdrant: QdrantClient, collection: str
) -> None:
    """Metadata enrichment must cost a payload update, not a re-embed."""
    encoder = HashTestEncoder()
    _seed_eliquis(qdrant, collection, encoder)
    target_id = point_id("eliquis:smpc:4.3", "en")
    before = qdrant.retrieve(
        collection_name=collection, ids=[target_id], with_payload=True, with_vectors=True
    )[0]
    assert isinstance(before.vector, dict)
    assert "active_substance" not in (before.payload or {})

    updated = set_medicine_metadata(
        qdrant,
        collection=collection,
        metadata_by_slug={"eliquis": {"active_substance": "apixaban", "atc_code": "B01AF02"}},
    )
    assert updated == {"eliquis": 3}
    after = qdrant.retrieve(
        collection_name=collection, ids=[target_id], with_payload=True, with_vectors=True
    )[0]
    assert after.payload is not None
    assert after.payload["active_substance"] == "apixaban"
    assert after.payload["atc_code"] == "B01AF02"
    assert isinstance(after.vector, dict) and isinstance(before.vector, dict)
    assert after.vector["dense"] == before.vector["dense"], "vectors must not change"
    assert collection_count(qdrant, collection=collection) == 4


def test_delete_medicine_points_removes_only_that_medicine(
    qdrant: QdrantClient, collection: str
) -> None:
    """A targeted rebuild must not delete another medicine's chunks."""
    encoder = HashTestEncoder()
    _seed_eliquis(qdrant, collection, encoder)
    assert delete_medicine_points(qdrant, collection=collection, slug="xarelto") == 1
    assert collection_count(qdrant, collection=collection) == 3
    assert (
        collection_count(
            qdrant, collection=collection, query_filter=build_filter(medicine=["xarelto"])
        )
        == 0
    )
    assert (
        collection_count(
            qdrant, collection=collection, query_filter=build_filter(medicine=["eliquis"])
        )
        == 3
    )


def test_require_index_fails_loudly_and_manifest_must_exist(tmp_path: Path) -> None:
    """Negative control: a missing index must raise, not return an empty result."""
    client = QdrantClient(":memory:")
    try:
        with pytest.raises(IndexMissingError, match="does not exist"):
            require_index(client, collection="absent", corpus_version=None)
    finally:
        client.close()
    with pytest.raises(IndexMissingError, match="manifest"):
        load_manifest(tmp_path / "manifest.json")


def test_require_index_rejects_a_partial_collection(qdrant: QdrantClient, collection: str) -> None:
    """A build that is still running (or died) must not be served as the full index."""
    _seed_eliquis(qdrant, collection, HashTestEncoder())
    require_index(qdrant, collection=collection, corpus_version=None, expected_points=4)
    with pytest.raises(IndexIncompleteError, match="4 points but the manifest records 8797"):
        require_index(qdrant, collection=collection, corpus_version=None, expected_points=8797)


def test_ready_is_false_while_the_collection_disagrees_with_the_manifest(
    qdrant: QdrantClient, collection: str, tmp_path: Path
) -> None:
    """``/readyz`` reflects the manifest count, not merely a non-empty collection."""
    encoder = HashTestEncoder()
    _seed_eliquis(qdrant, collection, encoder)
    settings = Settings(data_dir=tmp_path, qdrant_collection=collection)

    def manifest_with(total: int) -> None:
        write_manifest(
            settings.index_manifest_file,
            build_manifest(
                collection=collection,
                corpus_version="v1",
                corpus_sha256_value="0" * 64,
                encoder=encoder,
                dense_vector_name=DENSE,
                sparse_vector_name=SPARSE,
                chunk_counts={"eliquis:en": total},
                medicines=["eliquis"],
                languages=["en", "el"],
                missing=[],
            ),
        )

    service = RetrievalService(settings, client=qdrant, encoder=encoder)
    manifest_with(8797)
    state = service.ready()
    assert state.ready is False
    assert "4 points but the manifest records 8797" in state.detail
    manifest_with(4)
    assert service.ready().ready is True


def _drop_build_collections(qdrant: QdrantClient, served: str) -> None:
    """Remove the served alias and every ``<served>_<stamp>`` collection a build left."""
    if alias_target(qdrant, served) is not None:
        qdrant.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=served))
            ]
        )
    for entry in qdrant.get_collections().collections:
        if entry.name.startswith(f"{served}_"):
            qdrant.delete_collection(entry.name)


def _fake_corpus(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, sections: list[Section]
) -> None:
    """Stub the parse stage so ``build_index`` runs without PDFs or a tokenizer."""
    medicine = Medicine.model_construct(  # stub: skips the two-language document check
        slug="eliquis",
        name="Eliquis",
        active_substance="apixaban",
        epar_url="https://www.ema.europa.eu/en/medicines/human/EPAR/eliquis",
    )
    settings.medicines_file.parent.mkdir(parents=True, exist_ok=True)
    settings.medicines_file.write_text("medicines: []\n", encoding="utf-8")
    result = ParseResult(medicine=medicine, lang=Lang.EN, sections=sections, origin="parsed")
    monkeypatch.setattr(pipeline, "load_medicines", lambda _path: [medicine])
    monkeypatch.setattr(pipeline, "chunk_budget_for", lambda _encoder, _settings: _word_budget())
    monkeypatch.setattr(pipeline, "parse_corpus", lambda *_args, **_kwargs: ([result], []))


def _word_budget() -> ChunkBudget:
    return ChunkBudget(max_tokens=512, reserve_tokens=0, count_tokens=lambda t: len(t.split()))


def _eliquis_sections(count: int) -> list[Section]:
    return [
        _section(
            medicine="eliquis",
            lang=Lang.EN,
            doc=DocKind.SMPC,
            number=f"4.{n}",
            text=f"Eliquis section {n} text.",
        )
        for n in range(count)
    ]


def test_point_alias_migrates_a_plain_collection_then_swaps(
    qdrant: QdrantClient, collection: str
) -> None:
    """The first alias build replaces the old real collection; later ones swap targets."""
    encoder = HashTestEncoder()
    old, new = f"{collection}_a", f"{collection}_b"
    try:
        _seed_eliquis(qdrant, collection, encoder)  # pre-alias layout: a real collection
        _seed_eliquis(qdrant, old, encoder)
        assert point_alias(qdrant, alias=collection, collection=old) is None
        assert alias_target(qdrant, collection) == old
        assert collection_count(qdrant, collection=collection) == 4

        ensure_collection(
            qdrant,
            collection=new,
            dense_dim=encoder.dimension,
            dense_vector_name=DENSE,
            sparse_vector_name=SPARSE,
        )
        assert point_alias(qdrant, alias=collection, collection=new) == old
        assert alias_target(qdrant, collection) == new
        assert collection_count(qdrant, collection=collection) == 0
    finally:
        _drop_build_collections(qdrant, collection)


def test_build_index_swaps_the_alias_and_drops_the_old_build(
    qdrant: QdrantClient, collection: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full rebuild serves the previous index until the new one is complete."""
    settings = Settings(data_dir=tmp_path, qdrant_collection=collection)
    encoder = HashTestEncoder()
    try:
        _fake_corpus(monkeypatch, settings, _eliquis_sections(3))
        build_index(settings=settings, client=qdrant, encoder=encoder, progress=lambda _l: None)
        first = alias_target(qdrant, collection)
        assert first is not None

        _fake_corpus(monkeypatch, settings, _eliquis_sections(5))
        manifest = build_index(
            settings=settings, client=qdrant, encoder=encoder, progress=lambda _l: None
        )
        second = alias_target(qdrant, collection)
        assert second not in (None, first)
        assert not qdrant.collection_exists(first)
        assert manifest.collection == collection
        assert collection_count(qdrant, collection=collection) == manifest.total_chunks == 5
    finally:
        _drop_build_collections(qdrant, collection)


def test_failed_build_keeps_serving_the_old_index(
    qdrant: QdrantClient, collection: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An embedding failure drops the half-built collection and leaves the alias alone."""
    settings = Settings(data_dir=tmp_path, qdrant_collection=collection)
    encoder = HashTestEncoder()
    try:
        _fake_corpus(monkeypatch, settings, _eliquis_sections(3))
        build_index(settings=settings, client=qdrant, encoder=encoder, progress=lambda _l: None)
        served = alias_target(qdrant, collection)

        def boom(_texts: Sequence[str]) -> list[list[float]]:
            raise RuntimeError("encoder crashed")

        monkeypatch.setattr(encoder, "encode_passages", boom)
        with pytest.raises(RuntimeError, match="encoder crashed"):
            build_index(
                settings=settings, client=qdrant, encoder=encoder, progress=lambda _l: None
            )
        assert alias_target(qdrant, collection) == served
        assert collection_count(qdrant, collection=collection) == 3
        leftovers = [
            c.name
            for c in qdrant.get_collections().collections
            if c.name.startswith(f"{collection}_") and c.name != served
        ]
        assert leftovers == []
    finally:
        _drop_build_collections(qdrant, collection)


@pytest.mark.parametrize(
    "empty_parse",
    [False, True],
    ids=["zero-chunks", "no-parse-results"],
)
def test_empty_build_cannot_replace_the_served_index(
    qdrant: QdrantClient,
    collection: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty_parse: bool,
) -> None:
    """``--allow-missing`` with no usable PDFs must not swap the alias to an empty index."""
    settings = Settings(data_dir=tmp_path, qdrant_collection=collection)
    encoder = HashTestEncoder()
    try:
        _fake_corpus(monkeypatch, settings, _eliquis_sections(3))
        build_index(settings=settings, client=qdrant, encoder=encoder, progress=lambda _l: None)
        served = alias_target(qdrant, collection)
        assert served is not None

        _fake_corpus(monkeypatch, settings, [])
        if empty_parse:
            monkeypatch.setattr(
                pipeline, "parse_corpus", lambda *_args, **_kwargs: ([], ["eliquis:en"])
            )
        with pytest.raises(IndexBuildError, match="refusing to replace"):
            build_index(
                settings=settings,
                client=qdrant,
                encoder=encoder,
                allow_missing=True,
                progress=lambda _l: None,
            )

        assert alias_target(qdrant, collection) == served, "the served alias must not move"
        assert qdrant.collection_exists(served)
        assert collection_count(qdrant, collection=collection) == 3
    finally:
        _drop_build_collections(qdrant, collection)


def test_a_value_error_from_a_dependency_is_unavailable_not_bad_input(
    qdrant: QdrantClient, collection: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the caller's own input earns a 422; a library ``ValueError`` (e.g. a pydantic
    ``ValidationError`` inside the Qdrant client) is a server-side failure."""
    encoder = HashTestEncoder()
    _seed_eliquis(qdrant, collection, encoder)
    settings = Settings(data_dir=tmp_path, qdrant_collection=collection)
    write_manifest(
        settings.index_manifest_file,
        build_manifest(
            collection=collection,
            corpus_version="v1",
            corpus_sha256_value="0" * 64,
            encoder=encoder,
            dense_vector_name=DENSE,
            sparse_vector_name=SPARSE,
            chunk_counts={"eliquis:en": 4},
            medicines=["eliquis"],
            languages=["en", "el"],
            missing=[],
        ),
    )

    def broken(_texts: Sequence[str]) -> list[list[float]]:
        raise ValueError("response did not validate")

    monkeypatch.setattr(encoder, "encode_queries", broken)
    service = RetrievalService(settings, client=qdrant, encoder=encoder)
    with pytest.raises(RetrievalUnavailableError, match="did not validate"):
        service.search("bleeding")
    with pytest.raises(ValueError, match="must not be empty"):
        service.search("  ")


def test_a_query_that_starts_with_the_passage_prefix_is_searched(
    qdrant: QdrantClient, collection: str, tmp_path: Path
) -> None:
    """User text is not a passage; the encoder's prefix guard must not reject it."""
    encoder = HashTestEncoder()
    _seed_eliquis(qdrant, collection, encoder)
    settings = Settings(data_dir=tmp_path, qdrant_collection=collection)
    write_manifest(
        settings.index_manifest_file,
        build_manifest(
            collection=collection,
            corpus_version="v1",
            corpus_sha256_value="0" * 64,
            encoder=encoder,
            dense_vector_name=DENSE,
            sparse_vector_name=SPARSE,
            chunk_counts={"eliquis:en": 4},
            medicines=["eliquis"],
            languages=["en", "el"],
            missing=[],
        ),
    )
    service = RetrievalService(settings, client=qdrant, encoder=PrefixEnforcingEncoder(encoder))
    response = service.search("passage: active bleeding", lang="en")
    assert response.results[0].chunk_id.startswith("eliquis:smpc:4.3")


def test_concurrent_cold_requests_load_the_model_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FastAPI runs sync endpoints on a thread pool; a cold start must not load N models."""
    loads: list[int] = []

    def slow_load(**_kwargs: object) -> HashTestEncoder:
        loads.append(1)
        time.sleep(0.2)
        return HashTestEncoder()

    monkeypatch.setattr(E5Encoder, "load", staticmethod(slow_load))
    service = RetrievalService(Settings(data_dir=tmp_path), client=QdrantClient(":memory:"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        encoders = list(pool.map(lambda _i: service.encoder, range(8)))
    assert len(loads) == 1
    assert all(encoder is encoders[0] for encoder in encoders)
