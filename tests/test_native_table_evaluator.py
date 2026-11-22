from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments" / "rust-pdf"
sys.path.insert(0, str(EXPERIMENT_DIR))

import run_comparison as comparison  # noqa: E402
import run_native_tables as evaluator  # noqa: E402


def _table(rows: list[list[str]]) -> dict[str, object]:
    return {
        "page": 1,
        "table_index": 0,
        "bbox_pt": [0.0, 0.0, 100.0, 100.0],
        "rows": [list(row) for row in rows],
        "bolds": [],
        "lines": [" - ".join(cell for cell in row if cell) for row in rows],
    }


def _frequency_layouts() -> list[tuple[str, list[list[str]], tuple[str, ...]]]:
    return [
        (
            "inline-SOC-colon",
            [
                ["System Organ Class", "Frequency: Adverse reaction"],
                [
                    "Immune system disorders",
                    "Common: Hypersensitivity reaction with qualifier\nUncommon: Angioedema",
                ],
            ],
            ("common", "uncommon"),
        ),
        (
            "frequency-first-Greek-singular",
            [
                ["Κατηγορία/Οργανικό Σύστημα", "Ανεπιθύμητες ενέργειες"],
                ["Διαταραχές του ανοσοποιητικού", ""],
                ["Συχνή", "Αντίδραση με κλινικό προσδιορισμό"],
                ["Πολύ συχνή", "Δεύτερη αντίδραση"],
            ],
            ("common", "very common"),
        ),
        (
            "multirow-frequency-header-grid",
            [
                ["MedDRA System Organ Class", "", ""],
                ["", "Very common", "Common"],
                ["Nervous system disorders", "Dizziness", "Headache"],
            ],
            ("very common", "common"),
        ),
        (
            "multiline-borderless-inline",
            [
                [
                    "Skin and subcutaneous tissue disorders\nCommon:\nPruritus\n"
                    "Uncommon:\nPustular psoriasis"
                ]
            ],
            ("common", "uncommon"),
        ),
    ]


def test_frequency_layouts_have_positive_oracles_and_reject_label_mutations() -> None:
    for layout, rows, expected_labels in _frequency_layouts():
        reference = _table(rows)
        assert evaluator.table_frequency_format(reference) != "unsupported", layout
        source, _ = evaluator.column_frequency_associations(reference)
        assert len(source) == len(expected_labels), layout
        assert {key[2] for key in source} == set(expected_labels), layout

        baseline = evaluator.compare_table_sets(
            [reference], evaluator.reference_as_candidate([reference])
        )
        assert all(evaluator.acceptance_criteria(baseline, schema_valid=True).values()), layout

        candidate = evaluator.reference_as_candidate([reference])
        if layout == "multirow-frequency-header-grid":
            candidate[0]["rows"][2][1:3] = reversed(candidate[0]["rows"][2][1:3])
        elif layout == "frequency-first-Greek-singular":
            candidate[0]["rows"][2][0] = "Όχι συχνή"
        else:
            original = candidate[0]["rows"]
            changed = False
            for row in original:
                for column, cell in enumerate(row):
                    if "Common:" in cell:
                        row[column] = cell.replace("Common:", "Rare:", 1)
                        changed = True
                        break
                if changed:
                    break
            assert changed, layout

        rejected = evaluator.acceptance_criteria(
            evaluator.compare_table_sets([reference], candidate), schema_valid=True
        )
        assert not rejected["frequency_association_precision_at_least_0_95"], layout
        assert not rejected["frequency_association_recall_at_least_0_95"], layout
        assert not rejected["no_reaction_column_frequency_assignment_mismatches"], layout


def test_ordered_cell_comparison_preserves_word_order_and_clinical_qualifiers() -> None:
    original = evaluator.ordered_text_units("rash with insulin-related hypoglycaemia")
    assert original != evaluator.ordered_text_units("rash hypoglycaemia with insulin-related")
    assert original != evaluator.ordered_text_units("rash with insulin-related")
    assert evaluator.ordered_text_units("Υπερευαισθ ησίαγ") == evaluator.ordered_text_units(
        "Υπερευαισθησίαγ"
    )
    assert evaluator.ordered_text_units("0.5") != evaluator.ordered_text_units("05")

    reference = _table([["System Organ Class", "Frequency: Adverse reaction"], ["Skin", "Common: Rash with qualifier"]])
    candidate = evaluator.reference_as_candidate([reference])
    candidate[0]["rows"][1][1] = "Common: Rash"
    metrics = evaluator.compare_table_sets([reference], candidate)
    assert metrics["frequency_association"]["matched_associations"] == 0
    assert not evaluator.acceptance_criteria(metrics, schema_valid=True)[
        "frequency_association_recall_at_least_0_95"
    ]
    assert metrics["cell_content_recall"] < 1.0


def test_frequency_headers_accept_ratio_qualifiers_and_split_greek_words() -> None:
    assert evaluator._exact_frequency_label("Very\nCommon (≥1/10)") == "very common"
    assert evaluator._exact_frequency_label("Common (≥1/100 to <1/10)") == "common"
    assert evaluator._exact_frequency_label("Rare (≥1/10000 to <1/1000)") == "rare"
    assert evaluator._exact_frequency_label(
        "Not known (cannot be estimated from the available data)"
    ) == "not known"
    split_unknown = evaluator.frequency_labels_in("Μη γνωστής συχνότη τας")
    assert len(split_unknown) == 1
    assert evaluator.canonical_frequency_label(split_unknown[0]) == evaluator.canonical_frequency_label(
        "Μη γνωστής συχνότητας"
    )

    reference = _table(
        [
            [
                "MedDRA System Organ Class",
                "Very\nCommon (≥1/10)",
                "Common (≥1/100 to <1/10)",
                "Uncommon (≥1/1 000 to <1/100)",
                "Rare (≥1/10000 to <1/1000)",
                "Not known (cannot be estimated from the available data)",
            ],
            ["Infections", "Nasopharyngitis", "Pneumonia", "Sepsis", "Aspergillosis", ""],
        ]
    )
    assert evaluator.table_frequency_format(reference) == "labelled"
    associations, _ = evaluator.column_frequency_associations(reference)
    assert len(associations) == 4
    assert {key[2] for key in associations} == {
        "very common",
        "common",
        "uncommon",
        "rare",
    }


def test_row_labelled_header_skips_title_and_uses_reaction_column() -> None:
    reference = _table(
        [
            ["Πίνακας 4: Ανεπιθύμητες ενέργειες", "", ""],
            [
                "Κατηγορία/οργανικό σύστημα σύμφωνα με το MedDRA",
                "Ανεπιθύμητες ενέργειες",
                "Συχνότητα",
            ],
            ["Λοιμώξεις και παρασιτώσεις", "Ρινοφαρυγγίτιδα", "Συχνές"],
            ["", "Λοίμωξη του ανώτερου αναπνευστικού", "Μη γνωστής συχνότη τας"],
        ]
    )
    assert evaluator.table_frequency_format(reference) == "row-labelled"
    associations, _ = evaluator.column_frequency_associations(reference)
    assert len(associations) == 2
    assert {key[0] for key in associations} == {
        evaluator.ordered_text_units("Ρινοφαρυγγίτιδα"),
        evaluator.ordered_text_units("Λοίμωξη του ανώτερου αναπνευστικού"),
    }
    assert {key[2] for key in associations} == {"common", "not known"}


def test_headerless_frequency_continuation_does_not_use_soc_as_reaction() -> None:
    reference = _table(
        [
            ["Διαταραχές του νευρικού συστήματος", "Αίσθηση καύσου", "Συχνές"],
            ["Αγγειακές διαταραχές", "Ερυθρίαση", "Πολύ συχνές"],
            ["", "Εξάψεις", "Συχνές"],
        ]
    )
    assert evaluator.table_frequency_format(reference) == "row-labelled"
    associations, _ = evaluator.column_frequency_associations(reference)
    assert len(associations) == 3
    assert {key[0] for key in associations} == {
        evaluator.ordered_text_units("Αίσθηση καύσου"),
        evaluator.ordered_text_units("Ερυθρίαση"),
        evaluator.ordered_text_units("Εξάψεις"),
    }


def test_frequency_header_is_inherited_from_aligned_source_continuation() -> None:
    header = [
        "System Organ Class",
        "Very common",
        "Common",
        "Uncommon",
        "Rare",
        "Very rare",
    ]
    source_header = _table(
        [
            header,
            ["Skin disorders", "", "Pruritus", "", "", ""],
        ]
    )
    source_header["page"] = 1
    source_continuation = _table(
        [["Musculoskeletal disorders", "", "Back pain", "", "", "Lupus-like syndrome"]]
    )
    source_continuation["page"] = 2
    source_tables = [source_header, source_continuation]

    candidate_header = _table(
        [
            header,
            ["Skin disorders", "", "Pruritus", "", "", ""],
        ]
    )
    candidate_header["page"] = 1
    candidate_continuation = _table(
        [["Musculoskeletal disorders", "", "Back pain", "", "", "Lupus-like syndrome"]]
    )
    candidate_continuation["page"] = 2
    candidate_tables = [candidate_header, candidate_continuation]

    metrics = evaluator.compare_table_sets(source_tables, candidate_tables)
    criteria = evaluator.acceptance_criteria(metrics, schema_valid=True)
    assert metrics["column_diagnostics"]["source_inherited_frequency_headers"] == [
        {
            "page": 2,
            "table_index": 0,
            "source_header_page": 1,
            "source_header_table_index": 0,
            "frequency_columns": {
                1: "very common",
                2: "common",
                3: "uncommon",
                4: "rare",
                5: "very rare",
            },
        }
    ]
    assert metrics["column_diagnostics"]["reference_associations"] == 3
    assert metrics["column_diagnostics"]["matched_associations"] == 3
    assert criteria["no_reaction_column_frequency_assignment_mismatches"]

    wrong_column = _table(
        [
            [
                "Musculoskeletal disorders",
                "",
                "",
                "Back pain",
                "",
                "Lupus-like syndrome",
            ]
        ]
    )
    wrong_column["page"] = 2
    rejected = evaluator.compare_table_sets(
        source_tables, [candidate_header, wrong_column]
    )
    rejected_criteria = evaluator.acceptance_criteria(rejected, schema_valid=True)
    assert not rejected_criteria[
        "no_reaction_column_frequency_assignment_mismatches"
    ]


def test_preferred_term_header_is_a_positive_semantic_oracle() -> None:
    reference = _table(
        [
            ["Table 3: Adverse reactions", "", ""],
            ["System Organ Class", "Preferred term", "Frequency category"],
            ["Nervous system disorders", "Headache with aura", "Common"],
        ]
    )
    native = _table(
        [
            ["Table 3: Adverse reactions", "", ""],
            ["System Organ Class", "Preferred term", "Frequency category"],
            ["Nervous system disorders", "Headache with aura", "Common"],
        ]
    )
    metrics = evaluator.compare_table_sets([reference], [native])
    criteria = evaluator.acceptance_criteria(metrics, schema_valid=True)
    assert metrics["frequency_gate_evidence"]["positive_reference_associations"] == 1
    assert metrics["column_diagnostics"]["reference_associations"] == 1
    assert criteria["no_reaction_column_frequency_assignment_mismatches"]

    wrong_frequency = _table(
        [
            ["Table 3: Adverse reactions", "", ""],
            ["System Organ Class", "Preferred term", "Frequency category"],
            ["Nervous system disorders", "Headache with aura", "Rare"],
        ]
    )
    rejected = evaluator.compare_table_sets([reference], [wrong_frequency])
    assert not evaluator.acceptance_criteria(rejected, schema_valid=True)[
        "no_reaction_column_frequency_assignment_mismatches"
    ]



def test_repeated_all_grade_frequency_groups_ignore_severity_columns() -> None:
    rows = [
        [
            "System Organ Class",
            "Drug A preferred term / overall frequency",
            "Frequency grade 3 or higher",
            "Drug B preferred term / overall frequency",
            "Frequency grade 3 or higher",
        ],
        ["Blood disorders", "", "", "", ""],
        ["Thrombocytopenia", "Common (7%)", "0.6%", "Very common (18.5%)", "6.9%"],
        ["Neutropenia", "Common (6%)", "0.9%", "Very common (24.6%)", "13.4%"],
    ]
    reference = _table(rows)
    source, _ = evaluator.column_frequency_associations(reference)
    assert evaluator.table_frequency_format(reference) == "row-labelled"
    assert len(source) == 4
    assert {key[1] for key in source} == {1, 3}
    assert {key[2] for key in source} == {"common", "very common"}

    candidate = evaluator.reference_as_candidate([reference])
    baseline = evaluator.compare_table_sets([reference], candidate)
    assert all(
        evaluator.acceptance_criteria(baseline, schema_valid=True).values()
    )

    candidate[0]["rows"][2][1] = "Rare (7%)"
    candidate[0]["rows"][2][2] = "Common (0.6%)"
    rejected = evaluator.compare_table_sets([reference], candidate)
    rejected_criteria = evaluator.acceptance_criteria(rejected, schema_valid=True)
    assert not rejected_criteria["frequency_association_precision_at_least_0_95"]
    assert not rejected_criteria["no_reaction_column_frequency_assignment_mismatches"]


def test_empty_frequency_oracle_never_passes() -> None:
    criteria = evaluator.acceptance_criteria(
        evaluator.compare_table_sets([], []), schema_valid=True
    )
    assert not criteria["frequency_association_precision_at_least_0_95"]
    assert not criteria["frequency_association_recall_at_least_0_95"]
    assert not criteria["no_reaction_column_frequency_assignment_mismatches"]



def test_tagrisso_el_source_fixture_has_five_column_frequency_oracle() -> None:
    medicines = {
        medicine.slug: medicine
        for medicine in comparison.load_medicines(ROOT / "data" / "medicines.yaml")
    }
    fixtures, adjudications = evaluator.load_reference_evidence(medicines)
    fixture = fixtures["tagrisso-el-table-2-page-12"]
    assert fixture["source_pdf_sha256"] == medicines["tagrisso"].documents[
        comparison.Lang.EL
    ].sha256
    assert fixture["page"] == 12
    assert "five physical columns" in fixture["provenance"]
    assert len(fixture["rows"]) == 17
    assert all(len(row) == 5 for row in fixture["rows"])
    assert len(fixture["source_frequency_associations"]) == 16

    column_associations, _ = evaluator.column_frequency_associations(fixture)
    row_associations, _ = evaluator.frequency_associations([], [fixture])
    assert sum(column_associations.values()) == 16
    assert {key[1] for key in column_associations} == {1, 3}
    assert sum(row_associations.values()) == 16
    assert (
        evaluator.ordered_text_units("Επανενεργοποίηση ηπατίτιδας Βδ"),
        1,
        "not known",
    ) in column_associations

    raw_tables = [
        {
            "page": page,
            "table_index": 0,
            "bbox_pt": [0.0, 0.0, 1.0, 1.0],
            "rows": [],
            "lines": [],
        }
        for page in (12, 13, 14)
    ]
    corrected, evidence = evaluator.adjudicate_reference_tables(
        raw_tables, 14, adjudications[("tagrisso", "el")], fixtures
    )
    assert [table["page"] for table in corrected] == [13, 14, 12]
    assert len(evidence["excluded_pymupdf_tables"]) == 1
    assert evidence["source_fixtures"][0]["fixture_id"] == fixture["id"]
    source_fixture_table = corrected[2]
    candidate = evaluator.reference_as_candidate([source_fixture_table])
    candidate[0]["rows"][6][1] = "Σπάνια (7%)"
    rejected = evaluator.compare_table_sets([source_fixture_table], candidate)
    rejected_criteria = evaluator.acceptance_criteria(rejected, schema_valid=True)
    assert not rejected_criteria["frequency_association_precision_at_least_0_95"]
    assert not rejected_criteria["no_reaction_column_frequency_assignment_mismatches"]
    wrong_unknown_column = evaluator.reference_as_candidate([source_fixture_table])
    wrong_unknown_column[0]["rows"][3][1] = ""
    wrong_unknown_column[0]["rows"][3][3] = "Μη γνωστής συχνότητας"
    metrics = evaluator.compare_table_sets([source_fixture_table], wrong_unknown_column)
    assert not evaluator.acceptance_criteria(metrics, schema_valid=True)[
        "no_reaction_column_frequency_assignment_mismatches"
    ]


def test_pinned_tysabri_el_fixtures_are_complete_source_tables() -> None:
    medicines = {
        medicine.slug: medicine
        for medicine in comparison.load_medicines(ROOT / "data" / "medicines.yaml")
    }
    fixtures, adjudications = evaluator.load_reference_evidence(medicines)
    adjudication = adjudications[("tysabri", "el")]
    source_tables = [
        fixtures[fixture_id]
        for fixture_id in adjudication["fixture_ids"]
    ]
    assert [table["page"] for table in source_tables] == [13, 14]
    assert all(
        table["source_pdf_sha256"] == medicines["tysabri"].documents[comparison.Lang.EL].sha256
        for table in source_tables
    )
    assert all(table["provenance"].strip() for table in source_tables)
    assert [len(table["rows"]) for table in source_tables] == [13, 4]
    assert all(len(row) == 6 for table in source_tables for row in table["rows"])

    raw = [
        {"page": page, "table_index": 0, "bbox_pt": [0.0, 0.0, 1.0, 1.0], "rows": [], "lines": []}
        for page in (13, 14, 16)
    ]
    corrected, evidence = evaluator.adjudicate_reference_tables(
        raw, 16, adjudication, fixtures
    )
    assert len(corrected) == 2
    assert len(evidence["excluded_pymupdf_tables"]) == 3
    assert sum(
        sum(evaluator.column_frequency_associations(table)[0].values())
        for table in corrected
    ) > 0

    with pytest.raises(ValueError, match="incomplete"):
        evaluator.adjudicate_reference_tables(raw[:-1], 16, adjudication, fixtures)


def test_cached_only_requests_full_manifest_without_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    download_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def unexpected_download(*args: object, **kwargs: object) -> str:
        download_calls.append((args, kwargs))
        raise AssertionError("cached-only mode must not download PDFs")

    def offline_probe(_library: Path, offline: bool = False) -> tuple[Path, list[str]]:
        assert offline
        probe = tmp_path / "probe"
        probe.write_bytes(b"test probe")
        return probe, ["cargo", "build", "--offline"]

    monkeypatch.setattr(comparison, "ASSET_DIR", tmp_path)
    monkeypatch.setattr(comparison, "download_pinned", unexpected_download)
    monkeypatch.setattr(evaluator, "cached_pdfium", lambda _library: (tmp_path / "libpdfium.so", "pinned"))
    monkeypatch.setattr(evaluator, "build_probe", offline_probe)

    report = evaluator.run_evaluation(
        argparse.Namespace(
            all=False,
            cached_only=True,
            output=tmp_path / "report.json",
            pdfium_library=None,
        )
    )
    capsys.readouterr()
    assert not download_calls
    assert report["summary"]["documents_requested"] == 80
    assert report["summary"]["documents_analyzed"] == 0
    assert len(report["documents"]) == 80
    assert all(document["status"] == "ERROR" for document in report["documents"])
    assert "--cached-only" in report["reproduction_command"]


def test_malformed_frequency_column_mapping_has_no_positive_result() -> None:
    reference = _table(
        [
            ["MedDRA System Organ Class", "", ""],
            ["", "Very common", "Common"],
            ["Nervous system disorders", "Dizziness", "Headache"],
        ]
    )
    candidate = evaluator.reference_as_candidate([reference])
    candidate[0]["rows"][2][1], candidate[0]["rows"][2][2] = (
        candidate[0]["rows"][2][2],
        candidate[0]["rows"][2][1],
    )
    metrics = evaluator.compare_table_sets([reference], candidate)
    assert metrics["column_diagnostics"]["missing_reference_column_frequency_associations"] > 0
    assert metrics["column_diagnostics"]["unexpected_native_column_frequency_associations"] > 0
    assert not evaluator.acceptance_criteria(metrics, schema_valid=True)[
        "no_reaction_column_frequency_assignment_mismatches"
    ]


def test_frequency_first_unknown_label_is_data_not_a_header() -> None:
    table = _table(
        [
            ["Κατηγορία/Οργανικό Σύστημα", "Ανεπιθύμητες ενέργειες"],
            ["Μη γνωστής\nσυχνότη τας", "Αναπνευστική καταστολή"],
            ["Συχνές", "Ζάλη"],
        ]
    )
    assert evaluator.table_frequency_format(table) == "frequency-first"
    associations, _ = evaluator.column_frequency_associations(table)
    assert associations == {
        (evaluator.ordered_text_units("Αναπνευστική καταστολή"), 0, "not known"): 1,
        (evaluator.ordered_text_units("Ζάλη"), 0, "common"): 1,
    }
    headerless = _table([["Μη γνωστής συχνότητας", "Αναπνευστική καταστολή"]])
    assert evaluator.table_frequency_format(headerless) == "frequency-first"
    assert evaluator.column_frequency_associations(headerless)[0] == {
        (evaluator.ordered_text_units("Αναπνευστική καταστολή"), 0, "not known"): 1
    }


def test_pinned_rybelsus_split_rows_require_both_source_fragments() -> None:
    medicines = {
        medicine.slug: medicine
        for medicine in comparison.load_medicines(ROOT / "data" / "medicines.yaml")
    }
    fixtures, _ = evaluator.load_reference_evidence(medicines)
    source = []
    for page in (8, 9):
        fixture = fixtures[f"rybelsus-el-table-1-page-{page}"]
        assert fixture["source_pdf_sha256"] == medicines["rybelsus"].documents[
            comparison.Lang.EL
        ].sha256
        table = _table(fixture["rows"])
        table["page"] = page
        table["bbox_pt"] = fixture["bbox_pt"]
        source.append(table)

    assert source[0]["rows"][-1][1:3] == ["Υπογλυκαιμία, όταν"] * 2
    assert source[1]["rows"][1][1].startswith("χρησιμοποιείται με ινσουλίνη")
    assert "Μειωμένη όρεξη" in source[1]["rows"][1][2]
    candidate = evaluator.reference_as_candidate(source)
    matches = evaluator.match_tables(source, candidate)
    logical_source, _ = evaluator._frequency_scoring_tables(source, candidate, matches)
    associations, _ = evaluator.column_frequency_associations(logical_source[0])
    assert (
        evaluator.ordered_text_units(
            "Υπογλυκαιμία, όταν χρησιμοποιείται με ινσουλίνη ή σουλφονυλουρίαα"
        ),
        1,
        "very common",
    ) in associations
    assert (
        evaluator.ordered_text_units(
            "Υπογλυκαιμία, όταν χρησιμοποιείται με άλλα από του στόματος "
            "αντιδιαβητικά προϊόνταα Μειωμένη όρεξη"
        ),
        2,
        "common",
    ) in associations
    assert source[0]["rows"][-1][1] == "Υπογλυκαιμία, όταν"
    assert source[1]["rows"][1][1].startswith("χρησιμοποιείται")

    baseline = evaluator.compare_table_sets(source, candidate)
    assert baseline["column_diagnostics"]["reference_associations"] > 2
    assert all(evaluator.acceptance_criteria(baseline, schema_valid=True).values())

    missing_tail = evaluator.reference_as_candidate(source)
    missing_tail[1]["rows"][1][1] = ""
    swapped_tails = evaluator.reference_as_candidate(source)
    swapped_tails[1]["rows"][1][1:3] = reversed(swapped_tails[1]["rows"][1][1:3])
    missing_appetite = evaluator.reference_as_candidate(source)
    missing_appetite[1]["rows"][1][2] = missing_appetite[1]["rows"][1][2].replace(
        "\nΜειωμένη όρεξη", ""
    )
    wrong_header = evaluator.reference_as_candidate(source)
    wrong_header[1]["rows"][0][1:3] = reversed(wrong_header[1]["rows"][0][1:3])
    for changed in (missing_tail, swapped_tails, missing_appetite, wrong_header):
        metrics = evaluator.compare_table_sets(source, changed)
        assert not evaluator.acceptance_criteria(metrics, schema_valid=True)[
            "no_reaction_column_frequency_assignment_mismatches"
        ]


def test_pinned_lyrica_en_source_preserves_frequency_rows_and_soc() -> None:
    medicines = {
        medicine.slug: medicine
        for medicine in comparison.load_medicines(ROOT / "data" / "medicines.yaml")
    }
    fixtures, adjudications = evaluator.load_reference_evidence(medicines)
    adjudication = adjudications[("lyrica", "en")]
    assert adjudication["fixture_ids"] == [
        "lyrica-en-table-2-page-10",
        "lyrica-en-table-2-page-11",
    ]
    tables = [fixtures[fixture_id] for fixture_id in adjudication["fixture_ids"]]
    associations = sum(
        (evaluator.column_frequency_associations(table)[0] for table in tables),
        start=collections.Counter(),
    )
    assert sum(associations.values()) == 46
    assert (evaluator.ordered_text_units("Nasopharyngitis"), 0, "common") in associations
    assert (evaluator.ordered_text_units("Respiratory depression"), 0, "not known") in associations
    assert (
        evaluator.ordered_text_units("Nasopharyngitis Blood and lymphatic system disorders"),
        0,
        "common",
    ) not in associations


def test_remicade_source_preserves_unknown_frequency_and_physical_column() -> None:
    medicines = {
        medicine.slug: medicine
        for medicine in comparison.load_medicines(ROOT / "data" / "medicines.yaml")
    }
    fixtures, adjudications = evaluator.load_reference_evidence(medicines)
    fixture = fixtures["remicade-en-table-1-page-16"]
    assert fixture["id"] in adjudications[("remicade", "en")]["fixture_ids"]
    assert fixture["source_pdf_sha256"] == medicines["remicade"].documents[
        comparison.Lang.EN
    ].sha256
    reference = _table(fixture["rows"])
    association = (
        evaluator.ordered_text_units(
            "Post-procedural complication (including infectious and non-infectious complications)"
        ),
        0,
        "not known",
    )
    source, _ = evaluator.column_frequency_associations(reference)
    assert association in source
    candidate = evaluator.reference_as_candidate([reference])
    candidate[0]["rows"][-1] = ["", candidate[0]["rows"][-1][0]]
    metrics = evaluator.compare_table_sets([reference], candidate)
    assert not evaluator.acceptance_criteria(metrics, schema_valid=True)[
        "no_reaction_column_frequency_assignment_mismatches"
    ]

    greek = [
        fixtures[f"remicade-el-table-1-page-{page}"]
        for page in (16, 17, 18, 19)
    ]
    all_greek = sum(
        (evaluator.column_frequency_associations(table)[0] for table in greek),
        start=collections.Counter(),
    )
    assert sum(all_greek.values()) == 59
    reaction = "Μετεγχειρητική επιπλοκή (συμπεριλαμβανομένων λοιμωδών και μη-λοιμωδών επιπλοκών)"
    assert (evaluator.ordered_text_units(reaction), 0, "not known") in all_greek
    split = _table(
        [[
            "Κακώσεις, δηλητηριάσεις και επιπλοκές θεραπευτικών χειρισμών\n"
            "Μη γνωστής\nσυχνότη\nτας:\n" + reaction
        ]]
    )
    assert evaluator.column_frequency_associations(split)[0] == {
        (evaluator.ordered_text_units(reaction), 0, "not known"): 1
    }
