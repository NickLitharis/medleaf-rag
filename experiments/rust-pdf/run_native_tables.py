from __future__ import annotations

import argparse
import collections
import difflib
import json
import math
import os
import re
import shutil
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXPERIMENT_DIR / "native-table-results.json"
CASES_PATH = EXPERIMENT_DIR / "native-table-cases.json"
FULL_SAMPLE = ("xarelto", "eliquis", "humira")
LANGUAGES = ("en", "el")

# These gates are fixed before running the native parser. Exact row recall is a
# diagnostic, not a gate; character overlap alone cannot establish row/frequency parity.
TABLE_DETECTION_MIN_PRECISION = 0.90
TABLE_DETECTION_MIN_RECALL = 0.90
CELL_CONTENT_MIN_PRECISION = 0.95
CELL_CONTENT_MIN_RECALL = 0.95
FREQUENCY_MIN_PRECISION = 0.95
FREQUENCY_MIN_RECALL = 0.95
COLUMN_FREQUENCY_MIN_PRECISION = 0.95
COLUMN_FREQUENCY_MIN_RECALL = 0.95
TABLE_BBOX_IOU_MIN = 0.25

sys.path.insert(0, str(EXPERIMENT_DIR))
import run_comparison as comparison  # noqa: E402 - reuse its pinned helpers and production imports


FREQUENCY_LABELS = (
    "very common",
    "very rare",
    "not known",
    "uncommon",
    "common",
    "rare",
    "μη γνωστής συχνότητας",
    "μη γνωστή συχνότητα",
    "μη γνωστές",
    "μη γνωστής",
    "πολύ συχνές",
    "πολύ συχνή",
    "πολύ σπάνιες",
    "πολύ σπάνια",
    "όχι συχνές",
    "όχι συχνή",
    "συχνές",
    "συχνή",
    "σπάνιες",
    "σπάνια",
)
_NORMALIZED_FREQUENCY_LABELS = sorted(
    {comparison.normalize(label) for label in FREQUENCY_LABELS}, key=len, reverse=True
)


def _frequency_label_expression(label: str) -> str:
    return r"\s+".join(
        r"\s*".join(re.escape(character) for character in word)
        for word in label.split()
    )


_FREQUENCY_PATTERN = re.compile(
    r"(?<!\w)("
    + "|".join(_frequency_label_expression(label) for label in _NORMALIZED_FREQUENCY_LABELS)
    + r")(?!\w)"
)

def normalized_line(text: str) -> str:
    return " ".join(comparison.normalize(text).split())



def ordered_text_units(text: str) -> tuple[str, ...]:
    """Ignore layout whitespace while preserving every normalized character in order."""
    return tuple(character for character in comparison.normalize(text) if not character.isspace())


def canonical_frequency_label(label: str) -> str:
    normalized = normalized_line(label)
    compact = "".join(normalized.split())
    normalized = next(
        (
            known
            for known in _NORMALIZED_FREQUENCY_LABELS
            if "".join(known.split()) == compact
        ),
        normalized,
    )
    greek_equivalents = {
        "πολύ συχνή": "very common",
        "πολύ συχνές": "very common",
        "συχνή": "common",
        "συχνές": "common",
        "όχι συχνή": "uncommon",
        "όχι συχνές": "uncommon",
        "σπάνια": "rare",
        "σπάνιες": "rare",
        "πολύ σπάνια": "very rare",
        "πολύ σπάνιες": "very rare",
        "μη γνωστής συχνότητας": "not known",
        "μη γνωστή συχνότητα": "not known",
        "μη γνωστές": "not known",
        "μη γνωστής": "not known",
    }
    return {
        normalized_line(source): canonical
        for source, canonical in greek_equivalents.items()
    }.get(normalized, normalized)


def _line_frequency_pairs(line: str) -> list[tuple[str, str]]:
    """Read one flattened row without treating its SOC/group label as a reaction."""
    line = comparison.normalize(line)
    left, separator, right = line.partition(" - ")
    left = left.strip()
    if not separator:
        left, separator, right = line.partition(" – ")
        left, right = left.strip(), right.strip()
    else:
        right = right.strip()

    if separator:
        explicit_frequency = re.search(
            r"(?:frequency|συχνότητα)\s*:\s*([^;]+)",
            right,
            flags=re.IGNORECASE,
        )
        explicit_reaction = re.search(
            r"(?:adverse reaction|reaction|ανεπιθύμητη ενέργεια)\s*:\s*(.*)$",
            right,
            flags=re.IGNORECASE,
        )
        if explicit_frequency and explicit_reaction:
            labels = frequency_labels_in(explicit_frequency.group(1))
            if len(labels) == 1 and explicit_reaction.group(1).strip():
                return [(explicit_reaction.group(1).strip(), labels[0])]

        pairs = _inline_frequency_segments(right)
        if pairs:
            return pairs

        exact_labels = frequency_labels_in(right)
        if len(exact_labels) == 1 and canonical_frequency_label(right) == canonical_frequency_label(
            exact_labels[0]
        ):
            return [(left, exact_labels[0])] if left else []

        # Frequency-first line form: "Common - Reaction".
        left_labels = frequency_labels_in(left)
        if len(left_labels) == 1 and canonical_frequency_label(left) == canonical_frequency_label(
            left_labels[0]
        ):
            return [(right, left_labels[0])] if right else []
        return []

    # A frequency label in a cell is meaningful only when it leads a reaction
    # phrase. A later "common" in reaction prose is not an association.
    match = _FREQUENCY_PATTERN.match(line.strip())
    if match is None:
        return []
    reaction = line.strip()[match.end() :].strip(" :;\n\t")
    return [(reaction, match.group(1))] if reaction else []

def frequency_associations(
    lines: list[str],
    tables: list[dict[str, Any]] | None = None,
) -> tuple[collections.Counter[Any], dict[Any, list[str]]]:
    counts: collections.Counter[Any] = collections.Counter()
    examples: dict[Any, list[str]] = collections.defaultdict(list)

    def add(reaction: str, label: str, example: str) -> None:
        reaction_key = ordered_text_units(reaction)
        if reaction_key:
            key = (reaction_key, canonical_frequency_label(label))
            counts[key] += 1
            examples[key].append(example)

    if tables is not None:
        for table in tables:
            source_associations = table.get("source_frequency_associations")
            if isinstance(source_associations, list):
                for association in source_associations:
                    if isinstance(association, dict):
                        add(
                            str(association.get("reaction", "")),
                            str(association.get("frequency", "")),
                            str(association.get("source", "pinned source table fixture")),
                        )
                continue

            table_associations, _ = column_frequency_associations(table)
            if table_associations:
                for (reaction_key, _column, label), count in table_associations.items():
                    key = (reaction_key, canonical_frequency_label(label))
                    counts[key] += count
                    examples[key].extend([f"source table column association: {label}"] * count)
                continue
            for line in table.get("lines", []):
                if isinstance(line, str):
                    for reaction, label in _line_frequency_pairs(line):
                        add(reaction, label, line)
        return counts, examples

    for line in lines:
        for reaction, label in _line_frequency_pairs(line):
            add(reaction, label, line)
    return counts, examples


def frequency_labels_in(text: str) -> list[str]:
    normalized = normalized_line(text)
    return [
        normalized_line(match.group(1))
        for match in _FREQUENCY_PATTERN.finditer(normalized)
    ]


def frequency_label_regression_controls() -> dict[str, Any]:
    cases = (
        ("Very\ncommon", ["very common"]),
        ("Όχι\nσυχνές", ["όχι συχνέσ"]),
        ("Πολύ\nσυχνές", ["πολύ συχνέσ"]),
        ("Μη γνωστής\nσυχνότητας", ["μη γνωστήσ συχνότητασ"]),
    )
    results = [
        {"text": text, "expected": expected, "actual": frequency_labels_in(text)}
        for text, expected in cases
    ]
    return {
        "cases": results,
        "pass": all(case["actual"] == case["expected"] for case in results),
    }


def valid_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        coordinates = [float(number) for number in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(number) for number in coordinates):
        return None
    return coordinates


def reference_context(pdf_path: Path) -> dict[str, Any]:
    """Extract only the Python production reference, including its first-copy page owner."""
    raw_lines = comparison.extract_lines(str(pdf_path))
    raw_smpc = comparison.split_sections(raw_lines, doc=comparison.DocKind.SMPC)
    accepted_headings, copy_end = comparison.first_copy_headings(
        raw_lines, doc=comparison.DocKind.SMPC
    )
    owner_by_page = comparison._table_owner_pages(raw_lines, raw_smpc)
    owner_pages = sorted(page for page, owner in owner_by_page.items() if owner == "4.8")

    heading_records = []
    for hit in accepted_headings:
        if 0 <= hit.index < len(raw_lines):
            heading_records.append(
                {
                    "line_index": hit.index,
                    "page": raw_lines[hit.index].page,
                    "section_key": hit.key,
                    "text": raw_lines[hit.index].text,
                }
            )

    reference_tables: list[dict[str, Any]] = []
    with comparison.fitz.open(pdf_path) as document:
        page_count = document.page_count
        for page_number in owner_pages:
            if not 1 <= page_number <= page_count:
                continue
            page = document[page_number - 1]
            page_spans = comparison._page_bold_spans(page)
            for table_index, table in enumerate(page.find_tables().tables):
                rows = [[cell or "" for cell in row] for row in table.extract()]
                row_objects = getattr(table, "rows", None)
                bolds = (
                    comparison._cell_bold_flags(row_objects, page_spans)
                    if row_objects is not None
                    else None
                )
                bbox = valid_bbox(getattr(table, "bbox", None))
                reference_tables.append(
                    {
                        "page": page_number,
                        "table_index": table_index,
                        "bbox_pt": bbox,
                        "rows": rows,
                        "bolds": bolds,
                        # These are flattened by the existing Python production
                        # flattener for the reference side only.
                        "lines": comparison.flatten_table_rows(rows, bolds),
                    }
                )
        page_sizes = [(page.rect.width, page.rect.height) for page in document]

    return {
        "page_count": page_count,
        "page_sizes_pt": page_sizes,
        "raw_lines": raw_lines,
        "first_copy_section_keys": [number for number, _, _ in raw_smpc],
        "first_copy_accepted_headings": heading_records,
        "first_copy_end_line_index": copy_end,
        "owner_by_page": owner_by_page,
        "owner_pages_4_8": owner_pages,
        "tables": reference_tables,
    }


def native_tables(native_document: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Read the Rust tables/lines as-is; never infer or flatten a candidate row in Python."""
    errors: list[str] = []
    if not isinstance(native_document.get("engine"), str):
        errors.append("top-level engine is missing or is not a string")
    pages = native_document.get("pages")
    if not isinstance(pages, list):
        return [], errors + ["top-level pages is missing or is not a list"]

    tables: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            errors.append(f"page entry {page_index} is not an object")
            continue
        page_number = page.get("page", page_index)
        if not isinstance(page_number, int) or page_number < 1:
            errors.append(f"page entry {page_index} has an invalid page number")
            continue
        if "tables" not in page:
            errors.append(f"page {page_number} has no tables field")
            page_tables: Any = []
        else:
            page_tables = page["tables"]
        if not isinstance(page_tables, list):
            errors.append(f"page {page_number} tables field is not a list")
            continue

        for table_index, table in enumerate(page_tables):
            if not isinstance(table, dict):
                errors.append(f"page {page_number} table {table_index} is not an object")
                continue
            bbox = valid_bbox(table.get("bbox_pt"))
            if bbox is None:
                errors.append(f"page {page_number} table {table_index} has an invalid bbox_pt")

            raw_rows = table.get("rows")
            if not isinstance(raw_rows, list):
                errors.append(f"page {page_number} table {table_index} rows is not a list")
                raw_rows = []
            parsed_rows: list[list[str]] = []
            parsed_bolds: list[list[bool | None]] = []
            for row_index, row in enumerate(raw_rows):
                if not isinstance(row, dict) or not isinstance(row.get("cells"), list):
                    errors.append(
                        f"page {page_number} table {table_index} row {row_index} has no cells list"
                    )
                    continue
                row_texts: list[str] = []
                row_bolds: list[bool | None] = []
                for column_index, cell in enumerate(row["cells"]):
                    if cell is None:
                        row_texts.append("")
                        row_bolds.append(None)
                        continue
                    if not isinstance(cell, dict) or not isinstance(cell.get("text"), str):
                        errors.append(
                            f"page {page_number} table {table_index} cell {row_index}/{column_index} "
                            "is not null or an object with string text"
                        )
                        row_texts.append("")
                        row_bolds.append(None)
                        continue
                    if valid_bbox(cell.get("bbox_pt")) is None:
                        errors.append(
                            f"page {page_number} table {table_index} cell {row_index}/{column_index} "
                            "has an invalid bbox_pt"
                        )
                    bold = cell.get("bold")
                    if bold is not None and not isinstance(bold, bool):
                        errors.append(
                            f"page {page_number} table {table_index} cell {row_index}/{column_index} "
                            "bold is not boolean or null"
                        )
                        bold = None
                    row_texts.append(cell["text"])
                    row_bolds.append(bold)
                parsed_rows.append(row_texts)
                parsed_bolds.append(row_bolds)

            raw_lines = table.get("lines")
            if not isinstance(raw_lines, list):
                errors.append(f"page {page_number} table {table_index} lines is not a list")
                raw_lines = []
            lines: list[str] = []
            for line_index, line in enumerate(raw_lines):
                if not isinstance(line, str):
                    errors.append(
                        f"page {page_number} table {table_index} line {line_index} is not a string"
                    )
                    continue
                lines.append(line)

            tables.append(
                {
                    "page": page_number,
                    "table_index": table_index,
                    "bbox_pt": bbox,
                    "rows": parsed_rows,
                    "bolds": parsed_bolds,
                    "lines": lines,
                }
            )
    return tables, errors


def bbox_iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def match_tables(
    reference_tables: list[dict[str, Any]], native_tables: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Greedy one-to-one bbox matching, within page, for detection diagnostics only."""
    pairs: list[tuple[float, int, int]] = []
    for reference_index, reference in enumerate(reference_tables):
        for native_index, native in enumerate(native_tables):
            if reference["page"] != native["page"]:
                continue
            overlap = bbox_iou(reference["bbox_pt"], native["bbox_pt"])
            if overlap >= TABLE_BBOX_IOU_MIN:
                pairs.append((overlap, reference_index, native_index))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched_reference: set[int] = set()
    matched_native: set[int] = set()
    matches = []
    for overlap, reference_index, native_index in pairs:
        if reference_index in matched_reference or native_index in matched_native:
            continue
        matched_reference.add(reference_index)
        matched_native.add(native_index)
        matches.append(
            {
                "reference_index": reference_index,
                "native_index": native_index,
                "bbox_iou": overlap,
            }
        )
    return matches


def table_cell_texts(tables: list[dict[str, Any]]) -> list[str]:
    return [
        cell
        for table in tables
        for row in table["rows"]
        for cell in row
        if isinstance(cell, str) and cell.strip()
    ]



def frequency_metrics(
    reference_lines: list[str],
    native_lines: list[str],
    reference_tables: list[dict[str, Any]] | None = None,
    native_tables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reference, reference_examples = frequency_associations(reference_lines, reference_tables)
    native, native_examples = frequency_associations(native_lines, native_tables)
    matched = sum((reference & native).values())
    reference_total = sum(reference.values())
    native_total = sum(native.values())
    missing: list[str] = []
    extra: list[str] = []
    for key, count in (reference - native).items():
        missing.extend(reference_examples[key][:count])
    for key, count in (native - reference).items():
        extra.extend(native_examples[key][:count])
    return {
        "reference_associations": reference_total,
        "native_associations": native_total,
        "matched_associations": matched,
        "precision": matched / native_total if native_total else (1.0 if not reference_total else 0.0),
        "recall": matched / reference_total if reference_total else (1.0 if not native_total else 0.0),
        "missing_reference_rows": missing[:20],
        "unexpected_native_rows": extra[:20],
        "missing_association_count": sum((reference - native).values()),
        "unexpected_association_count": sum((native - reference).values()),
        "association_key": "whitespace-insensitive ordered NFKC/casefold reaction text units plus normalized frequency label",
    }


def _exact_frequency_label(text: str) -> str | None:
    normalized = normalized_line(text).strip()
    match = _FREQUENCY_PATTERN.match(normalized)
    if match is None:
        return None
    remainder = normalized[match.end() :].strip()
    if not remainder:
        return match.group(1)
    if re.fullmatch(r"\([^()]*\)", remainder) and not frequency_labels_in(remainder):
        return match.group(1)
    return None

def _header_role(rows: list[list[str]], markers: tuple[str, ...]) -> int | None:
    for column in range(max((len(row) for row in rows[:4]), default=0)):
        text = " ".join(row[column] for row in rows[:4] if column < len(row))
        normalized = normalized_line(text)
        if any(marker in normalized for marker in markers):
            return column
    return None


def _frequency_header(rows: list[list[str]]) -> tuple[dict[int, str], int]:
    frequency_columns: dict[int, str] = {}
    header_end = -1
    for row_index, row in enumerate(rows[:4]):
        labels = {
            column: label
            for column, cell in enumerate(row)
            if (label := _exact_frequency_label(cell)) is not None
        }
        has_frequency_heading = any(
            _is_frequency_header(cell) and _exact_frequency_label(cell) is None
            for cell in row
        )
        if labels and (len(labels) >= 2 or has_frequency_heading):
            frequency_columns = labels
            header_end = row_index
    return frequency_columns, header_end


def _inline_frequency_segments(text: str) -> list[tuple[str, str]]:
    normalized = comparison.normalize(text)
    matches = list(_FREQUENCY_PATTERN.finditer(normalized))
    valid = []
    for match in matches:
        before = normalized[: match.start()]
        if not before.strip() or before.rstrip(" \t").endswith((";", "\n")):
            valid.append(match)
    result = []
    for index, match in enumerate(valid):
        end = valid[index + 1].start() if index + 1 < len(valid) else len(normalized)
        reaction = normalized[match.end() : end].strip(" :;\n\t")
        if reaction:
            result.append((reaction, match.group(1)))
    return result


def _is_system_class_header(text: str) -> bool:
    normalized = normalized_line(text)
    return (
        "system organ class" in normalized
        or "meddra" in normalized
        or "κατηγορ" in normalized
        or ("οργανικ" in normalized and "σύστημα" in normalized)
    )


def _is_reaction_header(text: str) -> bool:
    normalized = normalized_line(text)
    return any(
        marker in normalized
        for marker in (
            "adverse reaction",
            "adverse drug reaction",
            "undesirable effect",
            "preferred term",
            "reaction",
            "ανεπιθύμητη ενέργεια",
            "ανεπιθύμητες ενέργειες",
            "ανεπιθύμητη αντίδραση",
            "ανεπιθύμητες αντιδράσεις",
        )
    )


def _is_frequency_header(text: str) -> bool:
    normalized = normalized_line(text)
    return "frequency" in normalized or "συχνότητα" in normalized


def _row_labelled_header(rows: list[list[str]]) -> tuple[int, int, int] | None:
    for row_index, row in enumerate(rows[:4]):
        frequency_columns = [
            column for column, cell in enumerate(row) if _is_frequency_header(cell)
        ]
        reaction_columns = [
            column for column, cell in enumerate(row) if _is_reaction_header(cell)
        ]
        if len(frequency_columns) == 1 and len(reaction_columns) == 1:
            frequency_column = frequency_columns[0]
            reaction_column = reaction_columns[0]
            if frequency_column != reaction_column:
                return frequency_column, reaction_column, row_index
    return None

def _repeated_overall_frequency_header(
    rows: list[list[str]],
) -> tuple[dict[int, str], int, int] | None:
    """Find repeated all-grade frequency columns beside one SOC/reaction column."""
    for header_end, row in enumerate(rows[:4]):
        frequency_columns = {
            column: "overall frequency"
            for column, cell in enumerate(row)
            if (
                (
                    ("overall" in normalized_line(cell) or "total" in normalized_line(cell))
                    and "frequency" in normalized_line(cell)
                )
                or (
                    "συνολικ" in normalized_line(cell)
                    and "συχνότη" in normalized_line(cell)
                )
            )
        }
        if len(frequency_columns) < 2:
            continue
        header_rows = rows[: header_end + 1]
        reaction_column = _header_role(
            header_rows, ("system organ class", "meddra", "κατηγορ")
        )
        if reaction_column is None:
            reaction_column = _header_role(
                header_rows,
                ("preferred term", "reaction", "όρος ευρετηρίου", "ανεπιθύμητη"),
            )
        if reaction_column is not None and reaction_column not in frequency_columns:
            return frequency_columns, reaction_column, header_end
    return None




def _unlabelled_frequency_columns(
    rows: list[list[str]],
) -> tuple[int, int] | None:
    labels_by_column: dict[int, list[int]] = collections.defaultdict(list)
    for row_index, row in enumerate(rows):
        for column, cell in enumerate(row):
            if _exact_frequency_label(cell) is not None:
                labels_by_column[column].append(row_index)

    candidates = []
    for frequency_column, labelled_rows in labels_by_column.items():
        if frequency_column < 2 or len(labelled_rows) < 2:
            continue
        reaction_column = frequency_column - 1
        reaction_rows = [
            row_index
            for row_index in labelled_rows
            if reaction_column < len(rows[row_index])
            and rows[row_index][reaction_column].strip()
            and _exact_frequency_label(rows[row_index][reaction_column]) is None
            and any(
                character.isalpha()
                for character in rows[row_index][reaction_column]
            )
        ]
        if len(reaction_rows) >= 2:
            candidates.append((len(reaction_rows), frequency_column, reaction_column))
    if not candidates:
        return None
    _, frequency_column, reaction_column = max(candidates)
    return frequency_column, reaction_column


def table_frequency_format(table: dict[str, Any]) -> str:
    rows = table["rows"]
    if not rows:
        return "unsupported"
    header = rows[0]
    if _repeated_overall_frequency_header(rows) is not None:
        return "row-labelled"
    frequency_columns, _ = _frequency_header(rows)
    if frequency_columns:
        if 0 in frequency_columns and _exact_frequency_label(header[0]) is not None:
            return "grid"
        return "labelled"
    inherited_context = table.get("_source_frequency_header_context")
    if inherited_context and inherited_context.get("frequency_columns"):
        return "labelled"
    if _row_labelled_header(rows) is not None or _unlabelled_frequency_columns(rows) is not None:
        return "row-labelled"
    if any(
        row
        and _exact_frequency_label(row[0]) is not None
        and any(cell.strip() for cell in row[1:])
        for row in rows
    ):
        return "frequency-first"
    if any(_inline_frequency_segments(cell) for row in rows for cell in row):
        return "inline"
    return "unsupported"


def _source_frequency_continuations(
    tables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Carry a pinned source grid header to the aligned, headerless next-page continuation."""
    annotated = [dict(table) for table in tables]
    ordered_indices = sorted(
        range(len(tables)),
        key=lambda index: (tables[index]["page"], tables[index]["table_index"]),
    )
    context: dict[str, Any] | None = None
    for index in ordered_indices:
        table = tables[index]
        page = table["page"]
        rows = table["rows"]
        frequency_columns, _ = _frequency_header(rows)
        if len(frequency_columns) >= 2:
            context = {
                "page": page,
                "table_index": table["table_index"],
                "column_count": max((len(row) for row in rows), default=0),
                "bbox_pt": valid_bbox(table.get("bbox_pt")),
                "frequency_columns": frequency_columns,
                "header_page": page,
                "header_table_index": table["table_index"],
            }
            continue

        if context is None:
            continue
        bbox = valid_bbox(table.get("bbox_pt"))
        previous_bbox = context["bbox_pt"]
        aligned = (
            bbox is not None
            and previous_bbox is not None
            and abs(bbox[0] - previous_bbox[0]) <= 3.0
            and abs(bbox[2] - previous_bbox[2]) <= 3.0
        )
        if (
            page != context["page"] + 1
            or max((len(row) for row in rows), default=0) != context["column_count"]
            or not aligned
            or table_frequency_format(table) != "unsupported"
        ):
            context = None
            continue

        frequency_columns = context["frequency_columns"]
        has_reaction_text = any(
            column < len(row)
            and row[column].strip()
            and any(character.isalpha() for character in row[column])
            for row in rows
            for column in frequency_columns
        )
        if not has_reaction_text:
            context = None
            continue

        inherited = {
            "header_page": context["header_page"],
            "header_table_index": context["header_table_index"],
            "frequency_columns": dict(frequency_columns),
        }
        annotated[index]["_source_frequency_header_context"] = inherited
        context = {
            **context,
            "page": page,
            "table_index": table["table_index"],
            "bbox_pt": bbox,
        }
    return annotated


def _join_frequency_row_continuations(
    tables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score a visibly continued row once, without changing either page's cells."""
    joined = [dict(table) for table in tables]
    ordered_indices = sorted(
        range(len(tables)),
        key=lambda index: (tables[index]["page"], tables[index]["table_index"]),
    )
    for previous_index, next_index in zip(ordered_indices, ordered_indices[1:]):
        previous, following = joined[previous_index], joined[next_index]
        if following["page"] != previous["page"] + 1:
            continue
        previous_bbox = valid_bbox(previous.get("bbox_pt"))
        next_bbox = valid_bbox(following.get("bbox_pt"))
        if (
            previous_bbox is None
            or next_bbox is None
            or abs(previous_bbox[0] - next_bbox[0]) > 3.0
            or abs(previous_bbox[2] - next_bbox[2]) > 3.0
        ):
            continue
        previous_rows, next_rows = previous["rows"], following["rows"]
        previous_columns, previous_header_end = _frequency_header(previous_rows)
        next_columns, next_header_end = _frequency_header(next_rows)
        if (
            len(previous_columns) < 2
            or {column: canonical_frequency_label(label) for column, label in previous_columns.items()}
            != {column: canonical_frequency_label(label) for column, label in next_columns.items()}
            or len(previous_rows) <= previous_header_end + 1
            or len(next_rows) <= next_header_end + 1
        ):
            continue
        previous_row = previous_rows[-1]
        next_row_index = next_header_end + 1
        next_row = next_rows[next_row_index]
        if (
            len(previous_row) != len(next_row)
            or not previous_row[0].strip()
            or not next_row[0].strip()
            or not next_row[0].lstrip()[0].islower()
            or not any(
                column < len(previous_row)
                and previous_row[column].strip()
                and next_row[column].strip()
                for column in previous_columns
            )
        ):
            continue
        # Retain both native fragments, including any later reaction in the
        # continuation cell. Only the comparison view joins the physical rows.
        combined = [
            " ".join(fragment.strip() for fragment in (left, right) if fragment.strip())
            for left, right in zip(previous_row, next_row)
        ]
        previous["rows"] = [*previous_rows[:-1], combined]
        following["rows"] = [
            *next_rows[:next_row_index],
            [""] * len(next_row),
            *next_rows[next_row_index + 1 :],
        ]
    return joined


def _frequency_scoring_tables(
    reference_tables: list[dict[str, Any]],
    native_tables: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_tables = _source_frequency_continuations(reference_tables)
    candidate_tables = [dict(table) for table in native_tables]
    for match in matches:
        source = source_tables[match["reference_index"]]
        context = source.get("_source_frequency_header_context")
        if context is not None:
            candidate_tables[match["native_index"]][
                "_source_frequency_header_context"
            ] = context
    return (
        _join_frequency_row_continuations(source_tables),
        _join_frequency_row_continuations(candidate_tables),
    )


def column_frequency_associations(
    table: dict[str, Any],
) -> tuple[collections.Counter[Any], dict[Any, list[str]]]:
    """Map ordered reaction text to the physical frequency/assignment column."""
    rows = table["rows"]
    counts: collections.Counter[Any] = collections.Counter()
    examples: dict[Any, list[str]] = collections.defaultdict(list)
    if not rows:
        return counts, examples

    def add(reaction: str, column: int, label: str, row_index: int, source: str) -> None:
        reaction_key = ordered_text_units(reaction)
        if not reaction_key:
            return
        key = (reaction_key, column, canonical_frequency_label(label))
        counts[key] += 1
        examples[key].append(
            f"row {row_index + 1}, column {column + 1}: {source} -> {label}"
        )

    repeated_header = _repeated_overall_frequency_header(rows)
    if repeated_header is not None:
        repeated_columns, reaction_column, header_end = repeated_header
        for row_index in range(header_end + 1, len(rows)):
            row = rows[row_index]
            if reaction_column >= len(row) or not row[reaction_column].strip():
                continue
            for column in repeated_columns:
                if column >= len(row):
                    continue
                for label in frequency_labels_in(row[column]):
                    add(row[reaction_column], column, label, row_index, row[column])
        return counts, examples
    frequency_columns, header_end = _frequency_header(rows)
    inherited_context = table.get("_source_frequency_header_context")
    inherited_columns = (
        inherited_context.get("frequency_columns", {}) if inherited_context else {}
    )
    inherited_header = not frequency_columns and bool(inherited_columns)
    if inherited_header:
        frequency_columns = inherited_columns
        header_end = -1
    first_header = " ".join(rows[0])
    system_class_column = (
        _header_role(rows, ("system organ class", "meddra", "κατηγορ"))
        if _is_system_class_header(first_header)
        else None
    )
    labelled_header = _row_labelled_header(rows)
    if labelled_header is not None:
        frequency_column, reaction_column, labelled_header_end = labelled_header
        header_end = max(header_end, labelled_header_end)
    else:
        inferred_columns = _unlabelled_frequency_columns(rows)
        if inferred_columns is None:
            frequency_column = None
            reaction_column = None
        else:
            frequency_column, reaction_column = inferred_columns

    if frequency_columns:
        grid_form = inherited_header or (
            0 in frequency_columns and _exact_frequency_label(rows[0][0]) is not None
        )
        for row_index in range(header_end + 1, len(rows)):
            row = rows[row_index]
            if grid_form or system_class_column is not None:
                for column, label in frequency_columns.items():
                    if column < len(row) and row[column].strip():
                        add(row[column], column, label, row_index, row[column])
                continue
            reaction = (
                row[reaction_column]
                if reaction_column is not None and reaction_column < len(row)
                else (row[0] if row else "")
            )
            for column, label in frequency_columns.items():
                if column < len(row) and row[column].strip() and reaction:
                    add(reaction, column, label, row_index, reaction)
        return counts, examples

    if frequency_column is not None and reaction_column is not None:
        for row_index in range(header_end + 1, len(rows)):
            row = rows[row_index]
            if frequency_column >= len(row) or reaction_column >= len(row):
                continue
            for label in frequency_labels_in(row[frequency_column]):
                add(row[reaction_column], frequency_column, label, row_index, row[reaction_column])
        if counts:
            return counts, examples

    for row_index, row in enumerate(rows):
        if row_index <= header_end or not row:
            continue
        leading_label = _exact_frequency_label(row[0])
        if leading_label is not None:
            for column, reaction in enumerate(row[1:], start=1):
                if reaction.strip():
                    add(reaction, 0, leading_label, row_index, reaction)
            continue
        for column, cell in enumerate(row):
            for reaction, label in _inline_frequency_segments(cell):
                add(reaction, column, label, row_index, reaction)
    return counts, examples


def column_diagnostics(
    reference_tables: list[dict[str, Any]],
    native_tables: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_associations: collections.Counter[Any] = collections.Counter()
    native_associations: collections.Counter[Any] = collections.Counter()
    reference_examples: dict[Any, list[str]] = collections.defaultdict(list)
    native_examples: dict[Any, list[str]] = collections.defaultdict(list)
    column_count_mismatches = []
    table_format_mismatches = []
    native_by_reference = {
        match["reference_index"]: match["native_index"] for match in matches
    }
    matched_native = {match["native_index"] for match in matches}

    for match in matches:
        reference = reference_tables[match["reference_index"]]
        native = native_tables[match["native_index"]]
        reference_columns = max((len(row) for row in reference["rows"]), default=0)
        native_columns = max((len(row) for row in native["rows"]), default=0)
        if reference_columns != native_columns:
            column_count_mismatches.append(
                {
                    "page": reference["page"],
                    "reference_table_index": reference["table_index"],
                    "native_table_index": native["table_index"],
                    "reference_columns": reference_columns,
                    "native_columns": native_columns,
                }
            )
        reference_format = table_frequency_format(reference)
        native_format = table_frequency_format(native)
        if reference_format != native_format:
            table_format_mismatches.append(
                {
                    "page": reference["page"],
                    "reference_table_index": reference["table_index"],
                    "native_table_index": native["table_index"],
                    "reference_format": reference_format,
                    "native_format": native_format,
                }
            )

    reference_format_counts: collections.Counter[str] = collections.Counter()
    native_format_counts: collections.Counter[str] = collections.Counter()
    applicable_reference_tables = 0
    unmatched_reference_tables = 0
    supported_formats = {"grid", "labelled", "row-labelled", "frequency-first", "inline"}

    for reference_index, reference in enumerate(reference_tables):
        reference_format = table_frequency_format(reference)
        ref_counts, ref_samples = column_frequency_associations(reference)
        if reference_format not in supported_formats or not ref_counts:
            continue
        applicable_reference_tables += 1
        reference_format_counts[reference_format] += 1
        reference_associations.update(ref_counts)
        for key, samples in ref_samples.items():
            reference_examples[key].extend(samples)

        native_index = native_by_reference.get(reference_index)
        if native_index is None:
            unmatched_reference_tables += 1
            continue
        native = native_tables[native_index]
        native_format = table_frequency_format(native)
        native_format_counts[native_format] += 1
        if native_format not in supported_formats:
            continue
        native_counts, native_samples = column_frequency_associations(native)
        native_associations.update(native_counts)
        for key, samples in native_samples.items():
            native_examples[key].extend(samples)

    for native_index, native in enumerate(native_tables):
        if native_index in matched_native:
            continue
        native_format = table_frequency_format(native)
        if native_format not in supported_formats:
            continue
        native_format_counts[native_format] += 1
        native_counts, native_samples = column_frequency_associations(native)
        native_associations.update(native_counts)
        for key, samples in native_samples.items():
            native_examples[key].extend(samples)

    matched_associations = sum((reference_associations & native_associations).values())
    reference_total = sum(reference_associations.values())
    native_total = sum(native_associations.values())
    missing = []
    extra = []
    for key, count in (reference_associations - native_associations).items():
        missing.extend(reference_examples[key][:count])
    for key, count in (native_associations - reference_associations).items():
        extra.extend(native_examples[key][:count])
    return {
        "reference_mapping_applicable": reference_total > 0,
        "mapping_applicability": (
            "applied only to source-supported grid, labelled, row-labelled, frequency-first, or inline layouts with positive ordered reaction/frequency associations; otherwise the separate row-level oracle must be positive"
        ),
        "reference_table_formats": dict(reference_format_counts),
        "native_table_formats": dict(native_format_counts),
        "applicable_reference_tables": applicable_reference_tables,
        "unmatched_applicable_reference_tables": unmatched_reference_tables,
        "reference_associations": reference_total,
        "native_associations": native_total,
        "matched_associations": matched_associations,
        "precision": (
            matched_associations / native_total
            if native_total
            else (0.0 if reference_total else None)
        ),
        "recall": matched_associations / reference_total if reference_total else None,
        "matched_tables_used": len(matches),
        "column_count_mismatch_count": len(column_count_mismatches),
        "column_count_mismatches": column_count_mismatches[:50],
        "table_format_mismatch_count": len(table_format_mismatches),
        "table_format_mismatches": table_format_mismatches[:50],
        "missing_reference_column_frequency_associations": sum(
            (reference_associations - native_associations).values()
        ),
        "unexpected_native_column_frequency_associations": sum(
            (native_associations - reference_associations).values()
        ),
        "missing_examples": missing[:20],
        "unexpected_examples": extra[:20],
        "source_inherited_frequency_headers": [
            {
                "page": table["page"],
                "table_index": table["table_index"],
                "source_header_page": table["_source_frequency_header_context"][
                    "header_page"
                ],
                "source_header_table_index": table[
                    "_source_frequency_header_context"
                ]["header_table_index"],
                "frequency_columns": table["_source_frequency_header_context"][
                    "frequency_columns"
                ],
            }
            for table in reference_tables
            if table.get("_source_frequency_header_context")
        ],
        "association_key": "whitespace-insensitive ordered NFKC/casefold reaction text units + physical zero-based frequency/assignment column + normalized frequency label",
    }


def compare_table_sets(
    reference_tables: list[dict[str, Any]], native_tables: list[dict[str, Any]]
) -> dict[str, Any]:
    reference_lines = [line for table in reference_tables for line in table["lines"]]
    native_lines = [line for table in native_tables for line in table["lines"]]
    reference_cells = table_cell_texts(reference_tables)
    native_cells = table_cell_texts(native_tables)

    reference_unit_count = sum(len(ordered_text_units(cell)) for cell in reference_cells)
    native_unit_count = sum(len(ordered_text_units(cell)) for cell in native_cells)
    matches = match_tables(reference_tables, native_tables)
    matched_units = 0
    for match in matches:
        reference_table = reference_tables[match["reference_index"]]
        native_table = native_tables[match["native_index"]]
        reference_sequence = tuple(
            token
            for row in reference_table["rows"]
            for cell in row
            for token in ordered_text_units(cell)
        )
        native_sequence = tuple(
            token
            for row in native_table["rows"]
            for cell in row
            for token in ordered_text_units(cell)
        )
        matcher = difflib.SequenceMatcher(
            a=reference_sequence, b=native_sequence, autojunk=False
        )
        matched_units += sum(block.size for block in matcher.get_matching_blocks())

    reference_rows = collections.Counter(normalized_line(line) for line in reference_lines if line.strip())
    native_rows = collections.Counter(normalized_line(line) for line in native_lines if line.strip())
    exact_rows = sum((reference_rows & native_rows).values())
    frequency_reference_tables, frequency_native_tables = _frequency_scoring_tables(
        reference_tables, native_tables, matches
    )
    frequency = frequency_metrics(
        reference_lines, native_lines, frequency_reference_tables, frequency_native_tables
    )
    column_frequency = column_diagnostics(
        frequency_reference_tables, frequency_native_tables, matches
    )
    row_frequency_applicable = frequency["reference_associations"] > 0
    column_frequency_applicable = column_frequency["reference_associations"] > 0
    if row_frequency_applicable:
        frequency_oracle = "ordered-source-row-frequency-associations"
        frequency_precision = frequency["precision"]
        frequency_recall = frequency["recall"]
        frequency_precision_pass = (
            frequency["native_associations"] > 0
            and frequency_precision >= FREQUENCY_MIN_PRECISION
        )
        frequency_recall_pass = (
            frequency["reference_associations"] > 0
            and frequency_recall >= FREQUENCY_MIN_RECALL
        )
    elif column_frequency_applicable:
        frequency_oracle = "positive-reference-reaction-column-frequency-associations"
        frequency_precision = column_frequency["precision"]
        frequency_recall = column_frequency["recall"]
        frequency_precision_pass = (
            column_frequency["native_associations"] > 0
            and frequency_precision is not None
            and frequency_precision >= COLUMN_FREQUENCY_MIN_PRECISION
        )
        frequency_recall_pass = (
            column_frequency["reference_associations"] > 0
            and frequency_recall is not None
            and frequency_recall >= COLUMN_FREQUENCY_MIN_RECALL
        )
    else:
        frequency_oracle = "missing-positive-reference-frequency-oracle"
        frequency_precision = None
        frequency_recall = None
        frequency_precision_pass = False
        frequency_recall_pass = False
    frequency_gate = {
        "oracle": frequency_oracle,
        "positive_reference_associations": max(
            frequency["reference_associations"],
            column_frequency["reference_associations"],
        ),
        "precision": frequency_precision,
        "recall": frequency_recall,
        "precision_pass": frequency_precision_pass,
        "recall_pass": frequency_recall_pass,
    }
    matched_table_count = len(matches)
    reference_table_count = len(reference_tables)
    native_table_count = len(native_tables)

    return {
        "reference_tables": reference_table_count,
        "native_tables": native_table_count,
        "matched_tables": matched_table_count,
        "table_detection_precision": (
            matched_table_count / native_table_count if native_table_count else 0.0
        ),
        "table_detection_recall": (
            matched_table_count / reference_table_count if reference_table_count else 0.0
        ),
        "table_bbox_iou_match_threshold": TABLE_BBOX_IOU_MIN,
        "table_bbox_matches": [
            {
                "page": reference_tables[match["reference_index"]]["page"],
                "reference_table_index": reference_tables[match["reference_index"]]["table_index"],
                "native_table_index": native_tables[match["native_index"]]["table_index"],
                "bbox_iou": round(match["bbox_iou"], 6),
            }
            for match in matches
        ],
        "reference_rows": sum(len(table["rows"]) for table in reference_tables),
        "native_rows": sum(len(table["rows"]) for table in native_tables),
        "reference_nonempty_cells": len(reference_cells),
        "native_nonempty_cells": len(native_cells),
        "reference_cell_ordered_text_units": reference_unit_count,
        "native_cell_ordered_text_units": native_unit_count,
        "matched_cell_ordered_text_units": matched_units,
        "cell_content_precision": (
            matched_units / native_unit_count if native_unit_count else 0.0
        ),
        "cell_content_recall": (
            matched_units / reference_unit_count if reference_unit_count else 0.0
        ),
        "reference_flattened_lines": len(reference_lines),
        "native_rust_flattened_lines": len(native_lines),
        "exact_normalized_row_matches": exact_rows,
        "exact_normalized_row_recall": exact_rows / sum(reference_rows.values()) if reference_rows else 1.0,
        "exact_normalized_row_precision": exact_rows / sum(native_rows.values()) if native_rows else 1.0,
        "frequency_association": frequency,
        "frequency_gate_evidence": frequency_gate,
        "column_diagnostics": column_frequency,
    }


def acceptance_criteria(metrics: dict[str, Any], schema_valid: bool) -> dict[str, bool]:
    frequency = metrics["frequency_association"]
    frequency_gate = metrics["frequency_gate_evidence"]
    column_frequency = metrics["column_diagnostics"]
    column_mapping_applicable = column_frequency["reference_mapping_applicable"]
    row_oracle_positive = frequency["reference_associations"] > 0
    row_oracle_pass = (
        row_oracle_positive
        and frequency["native_associations"] > 0
        and frequency["precision"] >= FREQUENCY_MIN_PRECISION
        and frequency["recall"] >= FREQUENCY_MIN_RECALL
    )
    column_precision_pass = (
        column_frequency["native_associations"] > 0
        and column_frequency["precision"] is not None
        and column_frequency["precision"] >= COLUMN_FREQUENCY_MIN_PRECISION
    )
    column_recall_pass = (
        column_frequency["reference_associations"] > 0
        and column_frequency["recall"] is not None
        and column_frequency["recall"] >= COLUMN_FREQUENCY_MIN_RECALL
    )
    return {
        "native_table_schema_valid": schema_valid,
        "reference_4_8_tables_present": metrics["reference_tables"] > 0,
        "native_table_detection_precision_at_least_0_90": (
            metrics["native_tables"] > 0
            and metrics["table_detection_precision"] >= TABLE_DETECTION_MIN_PRECISION
        ),
        "native_table_detection_recall_at_least_0_90": (
            metrics["reference_tables"] > 0
            and metrics["table_detection_recall"] >= TABLE_DETECTION_MIN_RECALL
        ),
        "cell_content_precision_at_least_0_95": (
            metrics["native_cell_ordered_text_units"] > 0
            and metrics["cell_content_precision"] >= CELL_CONTENT_MIN_PRECISION
        ),
        "cell_content_recall_at_least_0_95": (
            metrics["reference_cell_ordered_text_units"] > 0
            and metrics["cell_content_recall"] >= CELL_CONTENT_MIN_RECALL
        ),
        "frequency_association_precision_at_least_0_95": frequency_gate["precision_pass"],
        "frequency_association_recall_at_least_0_95": frequency_gate["recall_pass"],
        "reaction_column_frequency_precision_at_least_0_95_when_reference_mapping_is_supported": (
            column_precision_pass if column_mapping_applicable else row_oracle_pass
        ),
        "reaction_column_frequency_recall_at_least_0_95_when_reference_mapping_is_supported": (
            column_recall_pass if column_mapping_applicable else row_oracle_pass
        ),
        "no_reaction_column_frequency_assignment_mismatches": (
            (
                column_frequency["reference_associations"] > 0
                and column_frequency["missing_reference_column_frequency_associations"] == 0
                and column_frequency["unexpected_native_column_frequency_associations"] == 0
            )
            if column_mapping_applicable
            else row_oracle_pass
        ),
    }


def check_oracle_case(case: dict[str, Any], lines: list[str]) -> dict[str, Any]:
    check_results = []
    for check in case["checks"]:
        if "reaction_starts_with" in check:
            reaction = normalized_line(check["reaction_starts_with"])
            expected = collections.Counter(
                {
                    comparison.normalize(label): count
                    for label, count in check["frequency_labels"].items()
                }
            )
            matching_lines = []
            for line in lines:
                normalized = normalized_line(line)
                if not normalized.startswith(reaction):
                    continue
                actual = collections.Counter(frequency_labels_in(line))
                if actual == expected:
                    matching_lines.append(line)
            check_results.append(
                {
                    "expectation": check,
                    "pass": bool(matching_lines),
                    "matching_lines": matching_lines[:3],
                }
            )
        elif "not_glued_pair" in check:
            first, second = check["not_glued_pair"]
            bad = [
                line
                for line in lines
                if normalized_line(line).startswith(normalized_line(first))
                and normalized_line(second) in normalized_line(line)
            ]
            check_results.append(
                {"expectation": check, "pass": not bad, "violating_lines": bad[:3]}
            )
        elif "unbanded_soc_heading" in check:
            heading = normalized_line(check["unbanded_soc_heading"])
            bad = [
                line
                for line in lines
                if normalized_line(line).startswith(heading)
                and frequency_labels_in(line)
            ]
            check_results.append(
                {"expectation": check, "pass": not bad, "violating_lines": bad[:3]}
            )
        else:
            check_results.append({"expectation": check, "pass": False, "error": "unknown check"})
    return {
        "id": case["id"],
        "source_table": case["source_table"],
        "review_note": case.get("review_note"),
        "checks": check_results,
        "pass": all(item["pass"] for item in check_results),
    }


def reference_as_candidate(reference_tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """In-memory comparator control only; never used for native-parser scoring."""
    return [
        {
            "page": table["page"],
            "table_index": table["table_index"],
            "bbox_pt": table["bbox_pt"],
            "rows": [list(row) for row in table["rows"]],
            "bolds": [list(row) for row in table["bolds"]] if table["bolds"] else [],
            "lines": list(table["lines"]),
        }
        for table in reference_tables
    ]


def replace_frequency_suffix(line: str, old_label: str, new_label: str) -> str | None:
    reaction, separator, suffix = line.partition(" - ")
    if not separator:
        return None
    updated, count = re.subn(re.escape(old_label), new_label, suffix, count=1, flags=re.IGNORECASE)
    return f"{reaction} - {updated}" if count else None


def run_indication_column_swap_control(
    reference_tables: list[dict[str, Any]], case: dict[str, Any] | None
) -> dict[str, Any]:
    """Swap indication cells while preserving their frequency-label multiset."""
    source = (
        "In-memory copy of the pinned Eliquis EN 4.8 PyMuPDF reference table; "
        "only cell-column assignments change, no corpus files are modified."
    )
    if not reference_tables or case is None:
        return {"source": source, "pass": False, "error": "Eliquis EN reference/case unavailable"}

    baseline_candidate = reference_as_candidate(reference_tables)
    baseline_metrics = compare_table_sets(reference_tables, baseline_candidate)
    baseline_criteria = acceptance_criteria(baseline_metrics, schema_valid=True)
    precision_key = (
        "reaction_column_frequency_precision_at_least_0_95_when_reference_mapping_is_supported"
    )
    recall_key = (
        "reaction_column_frequency_recall_at_least_0_95_when_reference_mapping_is_supported"
    )
    baseline_column_gate = (
        baseline_criteria[precision_key]
        and baseline_criteria[recall_key]
        and baseline_criteria["no_reaction_column_frequency_assignment_mismatches"]
    )

    target = None
    for table_index, table in enumerate(reference_tables):
        if table_frequency_format(table) != "labelled":
            continue
        for row_index, row in enumerate(table["rows"][1:], start=1):
            if not row or not row[0].strip():
                continue
            frequency_cells = []
            for column, text in enumerate(row[1:], start=1):
                labels = frequency_labels_in(text)
                if len(labels) == 1:
                    frequency_cells.append((column, text, labels[0]))
            for first_index, first in enumerate(frequency_cells):
                second = next(
                    (
                        candidate
                        for candidate in frequency_cells[first_index + 1 :]
                        if candidate[2] != first[2]
                    ),
                    None,
                )
                if second is not None:
                    target = (table_index, row_index, row[0], first, second)
                    break
            if target is not None:
                break
        if target is not None:
            break

    if target is None:
        return {
            "source": source,
            "case_id": case["id"],
            "baseline_column_gate_passes": baseline_column_gate,
            "pass": False,
            "error": "no labelled reference row had two indication columns with distinct frequency labels",
        }

    table_index, row_index, reaction, first, second = target
    first_column, first_text, first_label = first
    second_column, second_text, second_label = second
    mutated_candidate = reference_as_candidate(reference_tables)
    mutated_row = mutated_candidate[table_index]["rows"][row_index]
    mutated_row[first_column], mutated_row[second_column] = second_text, first_text

    before_frequency_multiset = collections.Counter(
        label
        for text in reference_tables[table_index]["rows"][row_index][1:]
        for label in frequency_labels_in(text)
    )
    after_frequency_multiset = collections.Counter(
        label for text in mutated_row[1:] for label in frequency_labels_in(text)
    )
    baseline_frequency_metrics = baseline_metrics["frequency_association"]
    mutated_metrics = compare_table_sets(reference_tables, mutated_candidate)
    mutated_criteria = acceptance_criteria(mutated_metrics, schema_valid=True)
    mutated_frequency_metrics = mutated_metrics["frequency_association"]
    mutated_column_gate = (
        mutated_criteria[precision_key]
        and mutated_criteria[recall_key]
        and mutated_criteria["no_reaction_column_frequency_assignment_mismatches"]
    )
    failed_criteria = [name for name, passed in mutated_criteria.items() if not passed]
    headers = reference_tables[table_index]["rows"][0]
    return {
        "source": source,
        "case_id": case["id"],
        "source_table": case["source_table"],
        "table_format": "labelled",
        "reference_page": reference_tables[table_index]["page"],
        "reference_table_index": reference_tables[table_index]["table_index"],
        "reaction": reaction,
        "swapped_indication_columns": [
            {
                "column_1_based": first_column + 1,
                "header": headers[first_column] if first_column < len(headers) else None,
                "frequency_before": first_label,
                "frequency_after": second_label,
            },
            {
                "column_1_based": second_column + 1,
                "header": headers[second_column] if second_column < len(headers) else None,
                "frequency_before": second_label,
                "frequency_after": first_label,
            },
        ],
        "per_reaction_frequency_multiset_preserved": (
            before_frequency_multiset == after_frequency_multiset
        ),
        "candidate_flattened_lines_untouched": (
            reference_tables[table_index]["lines"] == mutated_candidate[table_index]["lines"]
        ),
        "baseline_column_metrics": baseline_metrics["column_diagnostics"],
        "mutated_column_metrics": mutated_metrics["column_diagnostics"],
        "baseline_column_gate_passes": baseline_column_gate,
        "mutated_column_gate_passes": mutated_column_gate,
        "row_frequency_metrics_changed": baseline_frequency_metrics != mutated_frequency_metrics,
        "mutated_acceptance_criteria": mutated_criteria,
        "mutated_failed_criteria": failed_criteria,
        "rejected_as_required": (
            baseline_column_gate
            and not mutated_column_gate
            and before_frequency_multiset == after_frequency_multiset
            and (
                mutated_metrics["column_diagnostics"][
                    "missing_reference_column_frequency_associations"
                ]
                > 0
                or mutated_metrics["column_diagnostics"][
                    "unexpected_native_column_frequency_associations"
                ]
                > 0
            )
        ),
        "pass": (
            baseline_column_gate
            and not mutated_column_gate
            and before_frequency_multiset == after_frequency_multiset
            and (
                mutated_metrics["column_diagnostics"][
                    "missing_reference_column_frequency_associations"
                ]
                > 0
                or mutated_metrics["column_diagnostics"][
                    "unexpected_native_column_frequency_associations"
                ]
                > 0
            )
        ),
    }


def frequency_layout_negative_controls() -> dict[str, Any]:
    layouts = [
        (
            "inline_SOC_colon",
            [
                ["System Organ Class", "Frequency: Adverse reaction"],
                [
                    "Immune system disorders",
                    "Common: Hypersensitivity reaction with qualifier\nUncommon: Angioedema",
                ],
            ],
            lambda table: table["rows"][1].__setitem__(
                1, table["rows"][1][1].replace("Common:", "Rare:", 1)
            ),
        ),
        (
            "frequency_first_Greek_singular",
            [
                ["Κατηγορία/Οργανικό Σύστημα", "Ανεπιθύμητες ενέργειες"],
                ["Διαταραχές του ανοσοποιητικού", ""],
                ["Συχνή", "Αντίδραση με κλινικό προσδιορισμό"],
                ["Πολύ συχνή", "Δεύτερη αντίδραση"],
            ],
            lambda table: table["rows"][2].__setitem__(0, "Όχι συχνή"),
        ),
        (
            "multirow_frequency_header_grid",
            [
                ["MedDRA System Organ Class", "", ""],
                ["", "Very common", "Common"],
                ["Nervous system disorders", "Dizziness", "Headache"],
            ],
            lambda table: table["rows"][2].__setitem__(
                slice(1, 3), list(reversed(table["rows"][2][1:3]))
            ),
        ),
        (
            "multiline_borderless_frequency_cell",
            [
                [
                    "Skin and subcutaneous tissue disorders\nCommon:\nPruritus\nUncommon:\nPustular psoriasis",
                ]
            ],
            lambda table: table["rows"][0].__setitem__(
                0, table["rows"][0][0].replace("Common:", "Rare:", 1)
            ),
        ),
    ]
    controls = []
    for name, rows, mutate in layouts:
        reference = {
            "page": 1,
            "table_index": 0,
            "bbox_pt": [0.0, 0.0, 100.0, 100.0],
            "rows": [list(row) for row in rows],
            "bolds": [],
            "lines": [" - ".join(cell for cell in row if cell) for row in rows],
        }
        baseline_candidate = reference_as_candidate([reference])
        baseline_metrics = compare_table_sets([reference], baseline_candidate)
        candidate = reference_as_candidate([reference])
        mutate(candidate[0])
        negative_metrics = compare_table_sets([reference], candidate)
        baseline_pass = all(acceptance_criteria(baseline_metrics, schema_valid=True).values())
        negative_criteria = acceptance_criteria(negative_metrics, schema_valid=True)
        rejected = not (
            negative_criteria["frequency_association_precision_at_least_0_95"]
            and negative_criteria["frequency_association_recall_at_least_0_95"]
            and negative_criteria["no_reaction_column_frequency_assignment_mismatches"]
        )
        controls.append(
            {
                "layout": name,
                "positive_baseline_passes": baseline_pass,
                "negative_frequency_mutation_rejected": rejected,
                "source_frequency_associations": baseline_metrics["frequency_association"][
                    "reference_associations"
                ],
                "negative_metrics": negative_metrics["frequency_association"],
                "pass": baseline_pass and rejected,
            }
        )
    return {
        "cases": controls,
        "pass": bool(controls) and all(control["pass"] for control in controls),
    }


def run_negative_controls(
    xarelto_reference_tables: list[dict[str, Any]],
    xarelto_case: dict[str, Any] | None,
    eliquis_en_reference_tables: list[dict[str, Any]],
    eliquis_en_case: dict[str, Any] | None,
) -> dict[str, Any]:
    """Exercise table, row-frequency, and indication-column gates on in-memory copies."""
    control_source = (
        "In-memory copies of pinned Xarelto EN and Eliquis EN 4.8 PyMuPDF reference tables; "
        "controls are not native-parser results and no corpus files are changed."
    )
    if not xarelto_reference_tables or xarelto_case is None:
        return {
            "source": control_source,
            "pass": False,
            "error": "Xarelto EN reference tables or its reviewed case were unavailable",
        }

    reference_lines = [line for table in xarelto_reference_tables for line in table["lines"]]
    baseline_candidate = reference_as_candidate(xarelto_reference_tables)
    baseline_metrics = compare_table_sets(xarelto_reference_tables, baseline_candidate)
    baseline_criteria = acceptance_criteria(baseline_metrics, schema_valid=True)
    baseline_pass = all(baseline_criteria.values())

    xarelto_expected = {
        check["reaction_starts_with"]: next(iter(check["frequency_labels"]))
        for check in xarelto_case["checks"]
        if "reaction_starts_with" in check
    }
    anaemia = next(
        (index for index, line in enumerate(reference_lines) if normalized_line(line).startswith("anaemia")),
        None,
    )
    thrombocytosis = next(
        (index for index, line in enumerate(reference_lines) if normalized_line(line).startswith("thrombocytosis")),
        None,
    )
    swap_reason = None
    swap_metrics = None
    swapped_lines = list(reference_lines)
    if anaemia is None or thrombocytosis is None:
        swap_reason = "the pinned reference did not contain both reviewed rows"
    else:
        first_old = xarelto_expected.get("Anaemia")
        second_old = xarelto_expected.get("Thrombocytosis")
        new_first = replace_frequency_suffix(
            reference_lines[anaemia], first_old or "", second_old or ""
        )
        new_second = replace_frequency_suffix(
            reference_lines[thrombocytosis], second_old or "", first_old or ""
        )
        if not new_first or not new_second or first_old == second_old:
            swap_reason = "could not make a distinct frequency swap in the reviewed reference rows"
        else:
            swapped_lines[anaemia] = new_first
            swapped_lines[thrombocytosis] = new_second
            swap_metrics = frequency_metrics(reference_lines, swapped_lines)

    missing_metrics = compare_table_sets(xarelto_reference_tables, [])
    missing_criteria = acceptance_criteria(missing_metrics, schema_valid=True)
    missing_rejected = not all(missing_criteria.values())
    empty_candidate = [
        {
            "page": table["page"],
            "table_index": table["table_index"],
            "bbox_pt": table["bbox_pt"],
            "rows": [],
            "bolds": [],
            "lines": [],
        }
        for table in xarelto_reference_tables
    ]
    empty_metrics = compare_table_sets(xarelto_reference_tables, empty_candidate)
    empty_criteria = acceptance_criteria(empty_metrics, schema_valid=True)
    empty_rejected = not all(empty_criteria.values())

    baseline_frequency_metrics = frequency_metrics(reference_lines, reference_lines)
    baseline_frequency_gate = (
        baseline_frequency_metrics["precision"] >= FREQUENCY_MIN_PRECISION
        and baseline_frequency_metrics["recall"] >= FREQUENCY_MIN_RECALL
    )
    swapped_frequency_gate = (
        swap_metrics is not None
        and swap_metrics["precision"] >= FREQUENCY_MIN_PRECISION
        and swap_metrics["recall"] >= FREQUENCY_MIN_RECALL
    )
    column_swap_control = run_indication_column_swap_control(
        eliquis_en_reference_tables, eliquis_en_case
    )
    label_controls = frequency_label_regression_controls()
    layout_controls = frequency_layout_negative_controls()
    return {
        "source": control_source,
        "baseline_row_frequency_metrics": baseline_frequency_metrics,
        "baseline_reference_copy_passes_all_gates": baseline_pass,
        "swapped_frequency_control": {
            "source_rows": [
                reference_lines[anaemia] if anaemia is not None else None,
                reference_lines[thrombocytosis] if thrombocytosis is not None else None,
            ],
            "mutated_rows": [
                replace_frequency_suffix(
                    reference_lines[anaemia],
                    xarelto_expected.get("Anaemia", ""),
                    xarelto_expected.get("Thrombocytosis", ""),
                )
                if anaemia is not None
                else None,
                replace_frequency_suffix(
                    reference_lines[thrombocytosis],
                    xarelto_expected.get("Thrombocytosis", ""),
                    xarelto_expected.get("Anaemia", ""),
                )
                if thrombocytosis is not None
                else None,
            ],
            "frequency_metrics": swap_metrics,
            "baseline_frequency_gate_passes": baseline_frequency_gate,
            "swapped_frequency_gate_passes": swapped_frequency_gate,
            "rejected_as_required": baseline_frequency_gate and not swapped_frequency_gate,
            "setup_error": swap_reason,
        },
        "missing_tables_control": {
            "candidate_tables_after_mutation": 0,
            "acceptance_criteria": missing_criteria,
            "rejected_as_required": missing_rejected,
        },
        "empty_tables_control": {
            "candidate_table_count": len(empty_candidate),
            "candidate_rows_after_mutation": 0,
            "candidate_lines_after_mutation": 0,
            "acceptance_criteria": empty_criteria,
            "rejected_as_required": empty_rejected,
        },
        "indication_column_frequency_swap_control": column_swap_control,
        "multiline_frequency_label_regression_controls": label_controls,
        "frequency_layout_negative_controls": layout_controls,
        "pass": (
            baseline_pass
            and swap_reason is None
            and baseline_frequency_gate
            and not swapped_frequency_gate
            and missing_rejected
            and empty_rejected
            and column_swap_control["pass"]
            and label_controls["pass"]
            and layout_controls["pass"]
        ),
    }


def load_cases(
    manifest_medicines: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    cases_document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if cases_document.get("schema_version") != 2 or not isinstance(
        cases_document.get("cases"), list
    ):
        raise ValueError(f"unsupported native-table cases file: {CASES_PATH}")
    cases = cases_document["cases"]
    by_document: dict[tuple[str, str], dict[str, Any]] = {}
    for case in cases:
        key = (case["medicine"], case["language"])
        if key in by_document:
            raise ValueError(f"duplicate native-table oracle case for {key}")
        if case["section"] != "4.8":
            raise ValueError(f"native-table oracle {case['id']} is not for section 4.8")
        medicine = manifest_medicines.get(case["medicine"])
        if medicine is None:
            raise ValueError(f"native-table oracle references unknown medicine {case['medicine']}")
        document = medicine.documents[comparison.Lang(case["language"])]
        if case["source_pdf_sha256"] != document.sha256:
            raise ValueError(
                f"oracle source hash for {key} differs from the pinned medicine manifest"
            )
        by_document[key] = case
    return cases, by_document


def load_reference_evidence(
    manifest_medicines: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    fixtures: dict[str, dict[str, Any]] = {}
    for fixture in document.get("source_reference_fixtures", []):
        fixture_id = fixture.get("id")
        if not isinstance(fixture_id, str) or fixture_id in fixtures:
            raise ValueError(f"invalid or duplicate source-reference fixture id: {fixture_id}")
        key = (fixture.get("medicine"), fixture.get("language"))
        medicine = manifest_medicines.get(key[0])
        if medicine is None:
            raise ValueError(f"source-reference fixture {fixture_id} references unknown medicine")
        source = medicine.documents[comparison.Lang(key[1])]
        if fixture.get("section") != "4.8" or fixture.get("source_pdf_sha256") != source.sha256:
            raise ValueError(f"source-reference fixture {fixture_id} does not match pinned section 4.8")
        if not isinstance(fixture.get("provenance"), str) or not fixture["provenance"].strip():
            raise ValueError(f"source-reference fixture {fixture_id} has no provenance")
        if not isinstance(fixture.get("page"), int) or fixture["page"] < 1:
            raise ValueError(f"source-reference fixture {fixture_id} has an invalid source page")
        if not isinstance(fixture.get("table_index"), int) or fixture["table_index"] < 0:
            raise ValueError(f"source-reference fixture {fixture_id} has an invalid table index")
        if valid_bbox(fixture.get("bbox_pt")) is None:
            raise ValueError(f"source-reference fixture {fixture_id} has an invalid source bbox")
        rows = fixture.get("rows")
        if (
            not isinstance(rows, list)
            or not rows
            or any(
                not isinstance(row, list)
                or not row
                or any(not isinstance(cell, str) for cell in row)
                for row in rows
            )
        ):
            raise ValueError(f"source-reference fixture {fixture_id} has invalid source rows")
        manual_associations = fixture.get("source_frequency_associations")
        fixture_table = {"rows": rows, "lines": []}
        if manual_associations is not None:
            if not isinstance(manual_associations, list) or not manual_associations:
                raise ValueError(
                    f"source-reference fixture {fixture_id} has invalid manual frequency associations"
                )
            for association in manual_associations:
                if (
                    not isinstance(association, dict)
                    or not isinstance(association.get("reaction"), str)
                    or not association["reaction"].strip()
                    or not isinstance(association.get("frequency"), str)
                    or len(frequency_labels_in(association["frequency"])) != 1
                    or not isinstance(association.get("source"), str)
                    or not association["source"].strip()
                ):
                    raise ValueError(
                        f"source-reference fixture {fixture_id} has invalid manual frequency associations"
                    )
            fixture_table["source_frequency_associations"] = manual_associations
        source_oracle, _ = frequency_associations([], [fixture_table])
        if not source_oracle:
            raise ValueError(f"source-reference fixture {fixture_id} has no positive frequency oracle")
        fixtures[fixture_id] = fixture

    adjudications: dict[tuple[str, str], dict[str, Any]] = {}
    for adjudication in document.get("reference_adjudications", []):
        key = (adjudication.get("medicine"), adjudication.get("language"))
        medicine = manifest_medicines.get(key[0])
        if medicine is None:
            raise ValueError(f"reference adjudication references unknown medicine {key[0]}")
        source = medicine.documents[comparison.Lang(key[1])]
        if (
            adjudication.get("section") != "4.8"
            or adjudication.get("source_pdf_sha256") != source.sha256
            or not isinstance(adjudication.get("provenance"), str)
            or not adjudication["provenance"].strip()
        ):
            raise ValueError(f"reference adjudication for {key} lacks pinned source provenance")
        if key in adjudications:
            raise ValueError(f"duplicate reference adjudication for {key}")
        for fixture_id in adjudication.get("fixture_ids", []):
            fixture = fixtures.get(fixture_id)
            if fixture is None or (fixture["medicine"], fixture["language"]) != key:
                raise ValueError(f"reference adjudication for {key} has invalid fixture {fixture_id}")
        adjudications[key] = adjudication
    return fixtures, adjudications


def adjudicate_reference_tables(
    raw_tables: list[dict[str, Any]],
    page_count: int,
    adjudication: dict[str, Any] | None,
    fixtures: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if adjudication is None:
        return raw_tables, {
            "applied": False,
            "raw_reference_table_count": len(raw_tables),
            "corrected_reference_table_count": len(raw_tables),
        }

    raw_by_id = {
        (table["page"], table["table_index"]): table
        for table in raw_tables
    }
    retained = adjudication.get("retain_pymupdf_tables", [])
    excluded = adjudication.get("exclude_pymupdf_tables", [])
    decisions: dict[tuple[int, int], tuple[str, str | None]] = {}
    for item in retained:
        key = (item.get("page"), item.get("table_index"))
        if key in decisions:
            raise ValueError(f"duplicate raw table decision {key}")
        decisions[key] = ("retained", None)
    for item in excluded:
        key = (item.get("page"), item.get("table_index"))
        reason = item.get("reason")
        if key in decisions or not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"invalid raw table exclusion for {key}")
        decisions[key] = ("excluded", reason)
    if set(decisions) != set(raw_by_id):
        missing = sorted(set(raw_by_id) - set(decisions))
        unknown = sorted(set(decisions) - set(raw_by_id))
        raise ValueError(
            f"raw reference table adjudication is incomplete; undecided={missing}, unknown={unknown}"
        )

    corrected = [
        table
        for table in raw_tables
        if decisions[(table["page"], table["table_index"])][0] == "retained"
    ]
    fixture_records = []
    for fixture_id in adjudication.get("fixture_ids", []):
        fixture = fixtures[fixture_id]
        page = fixture["page"]
        if page > page_count:
            raise ValueError(f"source-reference fixture {fixture_id} page exceeds the pinned PDF")
        rows = [list(row) for row in fixture["rows"]]
        fixture_lines = [
            " - ".join(cell.strip() for cell in row if cell.strip())
            for row in rows
            if any(cell.strip() for cell in row)
        ]
        corrected.append(
            {
                "page": page,
                "table_index": fixture["table_index"],
                "bbox_pt": valid_bbox(fixture["bbox_pt"]),
                "rows": rows,
                "bolds": None,
                "lines": fixture_lines,
                **(
                    {"source_frequency_associations": fixture["source_frequency_associations"]}
                    if fixture.get("source_frequency_associations")
                    else {}
                ),
            }
        )
        fixture_records.append(
            {
                "fixture_id": fixture_id,
                "source_page": page,
                "source_pdf_sha256": fixture["source_pdf_sha256"],
                "provenance": fixture["provenance"],
            }
        )
    retained_records = [
        {"page": page, "table_index": table_index}
        for (page, table_index), (decision, _) in sorted(decisions.items())
        if decision == "retained"
    ]
    excluded_records = [
        {
            "page": page,
            "table_index": table_index,
            "reason": reason,
        }
        for (page, table_index), (decision, reason) in sorted(decisions.items())
        if decision == "excluded"
    ]
    if not corrected:
        raise ValueError("reference adjudication removed all tables without a source fixture")
    return corrected, {
        "applied": True,
        "source_pdf_sha256": adjudication["source_pdf_sha256"],
        "provenance": adjudication["provenance"],
        "raw_reference_table_count": len(raw_tables),
        "retained_pymupdf_tables": retained_records,
        "excluded_pymupdf_tables": excluded_records,
        "source_fixtures": fixture_records,
        "corrected_reference_table_count": len(corrected),
    }


def cargo_metadata() -> dict[str, Any]:
    cargo_toml_path = EXPERIMENT_DIR / "Cargo.toml"
    cargo_lock_path = EXPERIMENT_DIR / "Cargo.lock"
    manifest = tomllib.loads(cargo_toml_path.read_text(encoding="utf-8"))
    lock = tomllib.loads(cargo_lock_path.read_text(encoding="utf-8"))
    locked = {package["name"]: package for package in lock.get("package", [])}
    package_metadata = manifest.get("package", {})
    dependency_names = sorted(manifest.get("dependencies", {}))
    crates = {}
    for name in dependency_names:
        package = locked.get(name)
        crates[name] = (
            {
                "version": package.get("version"),
                "registry_checksum": package.get("checksum"),
            }
            if package
            else {"version": None, "registry_checksum": None}
        )
    return {
        "application_package": {
            "name": package_metadata.get("name"),
            "version": package_metadata.get("version"),
            "edition": package_metadata.get("edition"),
        },
        "direct_dependency_lock_entries": crates,
        "cargo_toml_sha256": comparison.sha256_file(cargo_toml_path),
        "cargo_lock_sha256": comparison.sha256_file(cargo_lock_path),
    }


class ProbeOutputError(RuntimeError):
    def __init__(
        self, message: str, command: list[str], stdout: str, stderr: str, returncode: int
    ) -> None:
        super().__init__(message)
        self.cmd = command
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def run_native_probe(binary: Path, pdf_path: Path, library: Path) -> dict[str, Any]:
    command = [str(binary), str(pdf_path)]
    environment = os.environ.copy()
    environment["PDFIUM_LIBRARY_PATH"] = str(library)
    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode,
            command,
            output=process.stdout,
            stderr=process.stderr,
        )
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise ProbeOutputError(
            f"native probe emitted invalid JSON: {error}",
            command,
            process.stdout,
            process.stderr,
            process.returncode,
        ) from error
    if not isinstance(result, dict):
        raise ProbeOutputError(
            "native probe JSON root is not an object",
            command,
            process.stdout,
            process.stderr,
            process.returncode,
        )
    return result


def command_failure_details(error: Exception) -> dict[str, Any] | None:
    if not isinstance(error, (subprocess.CalledProcessError, ProbeOutputError)):
        return None

    def captured_output(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        limit = 20_000
        return {
            "text": text[-limit:],
            "truncated": len(text) > limit,
        }

    command = error.cmd
    if isinstance(command, (list, tuple)):
        command = [str(part) for part in command]
    else:
        command = str(command)
    return {
        "command": command,
        "returncode": error.returncode,
        "stdout": captured_output(error.stdout),
        "stderr": captured_output(error.stderr),
    }


def build_probe(library: Path, offline: bool = False) -> tuple[Path, list[str]]:
    cargo = comparison.ensure_cargo_binary()
    comparison.CARGO_HOME.mkdir(parents=True, exist_ok=True)
    comparison.TARGET_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CARGO_HOME"] = str(comparison.CARGO_HOME)
    environment["CARGO_TARGET_DIR"] = str(comparison.TARGET_DIR)
    environment["CARGO_BUILD_JOBS"] = "1"
    environment["PDFIUM_LIBRARY_PATH"] = str(library)
    environment["PATH"] = f"{cargo.parent}:{environment.get('PATH', '')}"
    command = [
        str(cargo),
        "build",
        "--release",
        "--locked",
        "--manifest-path",
        str(EXPERIMENT_DIR / "Cargo.toml"),
    ]
    if offline:
        command.append("--offline")
    subprocess.run(command, cwd=ROOT, env=environment, check=True, capture_output=True, text=True)
    binary = comparison.TARGET_DIR / "release/medleaf-ema-pdfium-probe"
    if not binary.is_file():
        raise RuntimeError(f"cargo build completed without expected binary: {binary}")
    return binary, command


def document_analysis(
    medicine: Any,
    language: Any,
    pdf_path: Path,
    pdf_hash: str,
    native_document: dict[str, Any],
    reference: dict[str, Any],
    case: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_native_tables, schema_errors = native_tables(native_document)
    owner_pages = set(reference["owner_pages_4_8"])
    selected_native_tables = [table for table in all_native_tables if table["page"] in owner_pages]
    outside_owner_pages = [table for table in all_native_tables if table["page"] not in owner_pages]
    raw_reference_tables = reference["tables"]
    reference_tables = reference.get("adjudicated_tables", raw_reference_tables)
    raw_metrics = compare_table_sets(raw_reference_tables, selected_native_tables)
    raw_criteria = acceptance_criteria(raw_metrics, schema_valid=not schema_errors)
    metrics = compare_table_sets(reference_tables, selected_native_tables)
    criteria = acceptance_criteria(metrics, schema_valid=not schema_errors)
    reference_lines = [line for table in reference_tables for line in table["lines"]]
    native_lines = [line for table in selected_native_tables for line in table["lines"]]

    oracle_result = None
    if case is not None:
        oracle_result = {
            "source_pdf_sha256": case["source_pdf_sha256"],
            "case_definition": case["source_table"],
            "reference": check_oracle_case(case, reference_lines),
            "native": check_oracle_case(case, native_lines),
        }
        oracle_result["pass"] = oracle_result["reference"]["pass"] and oracle_result["native"]["pass"]

    xarelto_result = None
    false_positive_soc_lines = []
    if medicine.slug == "xarelto" and language is comparison.Lang.EN:
        independent_reference = comparison.xarelto_oracle(reference_lines)
        independent_native = comparison.xarelto_oracle(native_lines)
        false_positive_soc_lines = [
            line
            for line in native_lines
            if any(
                normalized_line(line).startswith(normalized_line(heading))
                and any(f" - {frequency}" in line.casefold() for frequency in ("common", "rare"))
                for heading in comparison.XARELTO_SOC_HEADINGS
            )
        ]
        xarelto_result = {
            "oracle_source": "tests/test_m1_acceptance.py:208-264 (the existing eight independent Xarelto EN assertions)",
            "reference": independent_reference,
            "native": independent_native,
            "independent_check_count": len(independent_native["checks"]),
            "false_positive_bold_soc_headings": {
                "count": len(false_positive_soc_lines),
                "lines": false_positive_soc_lines[:10],
            },
            "pass": (
                len(independent_native["checks"]) == 8
                and independent_reference["pass"]
                and independent_native["pass"]
            ),
        }

    with comparison.fitz.open(pdf_path) as document:
        native_page_count = len(native_document.get("pages", []))
        first_copy_headings = reference["first_copy_accepted_headings"]
        result = {
            "medicine": medicine.slug,
            "language": language.value,
            "source_url": str(medicine.documents[language].url),
            "source_attribution": comparison.SOURCE_ATTRIBUTION,
            "manifest_pdf_sha256": medicine.documents[language].sha256,
            "pdf_sha256": pdf_hash,
            "page_count_pymupdf": document.page_count,
            "page_count_native": native_page_count,
            "native_engine": native_document.get("engine"),
            "python_reference_first_copy_selection": {
                "document": "SmPC",
                "section": "4.8",
                "selection_source": "Existing Python production extract_lines -> split_sections -> first_copy_headings -> _table_owner_pages; table pages are assigned to 4.8 in the first-copy span.",
                "first_copy_section_keys": reference["first_copy_section_keys"],
                "accepted_heading_count": len(first_copy_headings),
                "accepted_headings": first_copy_headings,
                "first_copy_end_line_index": reference["first_copy_end_line_index"],
                "owned_pages_4_8": reference["owner_pages_4_8"],
                "excluded_native_tables_outside_owned_pages": len(outside_owner_pages),
            },
            "native_table_schema_errors": schema_errors[:100],
            "native_tables_outside_python_4_8_pages": len(outside_owner_pages),
            "comparison": raw_metrics,
            "raw_criteria": raw_criteria,
            "adjudicated_comparison": metrics,
            "reference_table_adjudication": reference.get("table_adjudication"),
            "criteria": criteria,
            "oracle_case": oracle_result,
            "xarelto_independent_4_8_oracle": xarelto_result,
        }
    if oracle_result is not None:
        result["criteria"]["reviewed_source_table_oracle_pass"] = oracle_result["pass"]
    if xarelto_result is not None:
        result["criteria"]["existing_xarelto_en_eight_checks_pass"] = xarelto_result["pass"]
    result["acceptance_pass"] = all(result["criteria"].values())
    return result, reference_tables


def write_result(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
def cached_pdfium(library_arg: str | None) -> tuple[Path, str]:
    library = (
        Path(library_arg).resolve()
        if library_arg
        else comparison.ASSET_DIR / "pdfium-8125" / "libpdfium.so"
    )
    actual = comparison.sha256_file(library)
    if actual != comparison.PDFIUM_LIBRARY_SHA256:
        raise ValueError(
            f"cached PDFium library hash mismatch: expected "
            f"{comparison.PDFIUM_LIBRARY_SHA256}, got {actual}"
        )
    return library, actual


def verify_cached_pdf(pdf_path: Path, expected_sha256: str) -> str:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"cached-only mode: pinned PDF is not cached: {pdf_path}")
    actual = comparison.sha256_file(pdf_path)
    if actual != expected_sha256:
        raise ValueError(
            f"cached PDF hash mismatch: expected {expected_sha256}, got {actual}: {pdf_path}"
        )
    with pdf_path.open("rb") as source:
        if source.read(5) != b"%PDF-":
            raise ValueError(f"cached asset is not a PDF: {pdf_path}")
    return actual


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = ROOT / "data/medicines.yaml"
    manifest_hash = comparison.sha256_file(manifest_path)
    if manifest_hash != comparison.MANIFEST_SHA256:
        raise ValueError(
            f"manifest changed: expected {comparison.MANIFEST_SHA256}, got {manifest_hash}; refusing to repin"
        )
    medicines = comparison.load_medicines(manifest_path)
    by_slug = {medicine.slug: medicine for medicine in medicines}
    cached_only = bool(getattr(args, "cached_only", False))
    selected_medicines = (
        medicines
        if args.all or cached_only
        else [by_slug[slug] for slug in FULL_SAMPLE]
    )
    cases, cases_by_document = load_cases(by_slug)
    fixtures_by_id, adjudications_by_document = load_reference_evidence(by_slug)

    library, library_hash = (
        cached_pdfium(args.pdfium_library)
        if cached_only
        else comparison.ensure_pdfium(args.pdfium_library)
    )
    binary, build_command = build_probe(library, offline=cached_only)
    pdf_records = []
    document_results = []
    references_by_document: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for medicine in selected_medicines:
        for language in (comparison.Lang.EN, comparison.Lang.EL):
            document_ref = medicine.documents[language]
            if not document_ref.sha256:
                document_results.append(
                    {
                        "medicine": medicine.slug,
                        "language": language.value,
                        "status": "ERROR",
                        "error": "manifest has no pinned PDF SHA-256",
                    }
                )
                continue
            pdf_path = comparison.ASSET_DIR / f"{medicine.slug}_{language.value}.pdf"
            record: dict[str, Any] = {
                "medicine": medicine.slug,
                "language": language.value,
                "source_url": str(document_ref.url),
                "manifest_pdf_sha256": document_ref.sha256,
                "path": str(pdf_path),
            }
            reference = None
            try:
                if cached_only:
                    actual_digest = verify_cached_pdf(pdf_path, document_ref.sha256)
                else:
                    digest = comparison.download_pinned(
                        str(document_ref.url), pdf_path, document_ref.sha256
                    )
                    actual_digest = comparison.sha256_file(pdf_path)
                    if digest != document_ref.sha256 or actual_digest != document_ref.sha256:
                        raise ValueError(
                            f"PDF hash mismatch for {medicine.slug}/{language.value}: "
                            f"manifest {document_ref.sha256}, download {digest}, "
                            f"on-disk {actual_digest}"
                        )
                    with pdf_path.open("rb") as source:
                        if source.read(5) != b"%PDF-":
                            raise ValueError(f"pinned asset is not a PDF: {pdf_path}")
                record["actual_pdf_sha256"] = actual_digest
                record["verified"] = True
                pdf_records.append(record)

                reference = reference_context(pdf_path)
                key = (medicine.slug, language.value)
                corrected_tables, table_adjudication = adjudicate_reference_tables(
                    reference["tables"],
                    reference["page_count"],
                    adjudications_by_document.get(key),
                    fixtures_by_id,
                )
                reference["adjudicated_tables"] = corrected_tables
                reference["table_adjudication"] = table_adjudication
                references_by_document[key] = corrected_tables
                native_document = run_native_probe(binary, pdf_path, library)
                result, reference_tables = document_analysis(
                    medicine,
                    language,
                    pdf_path,
                    actual_digest,
                    native_document,
                    reference,
                    cases_by_document.get((medicine.slug, language.value)),
                )
                references_by_document[(medicine.slug, language.value)] = reference_tables
                result["status"] = "PASS" if result["acceptance_pass"] else "FAIL"
                document_results.append(result)
                table_metrics = result["adjudicated_comparison"]
                print(
                    f"{medicine.slug}/{language.value}: "
                    f"4.8 pages={len(reference['owner_pages_4_8'])} "
                    f"tables={table_metrics['native_tables']}/{table_metrics['reference_tables']} "
                    f"content P/R={table_metrics['cell_content_precision']:.4f}/"
                    f"{table_metrics['cell_content_recall']:.4f} "
                    f"frequency P/R={table_metrics['frequency_association']['precision']:.4f}/"
                    f"{table_metrics['frequency_association']['recall']:.4f} "
                    f"status={result['status']}"
                )
            except Exception as error:
                if record not in pdf_records and record.get("verified"):
                    pdf_records.append(record)
                failure = {
                    "medicine": medicine.slug,
                    "language": language.value,
                    "source_url": str(document_ref.url),
                    "manifest_pdf_sha256": document_ref.sha256,
                    "pdf_sha256": record.get("actual_pdf_sha256"),
                    "page_count_pymupdf": reference.get("page_count") if reference else None,
                    "python_reference_first_copy_selection": (
                        {
                            "document": "SmPC",
                            "section": "4.8",
                            "selection_source": "Existing Python production extract_lines -> split_sections -> first_copy_headings -> _table_owner_pages.",
                            "owned_pages_4_8": reference["owner_pages_4_8"],
                        }
                        if reference
                        else None
                    ),
                    "status": "ERROR",
                    "error": f"{type(error).__name__}: {error}",
                }
                command_failure = command_failure_details(error)
                if command_failure is not None:
                    failure["command_failure"] = command_failure
                    stderr_text = (command_failure.get("stderr") or {}).get("text")
                    if stderr_text:
                        print(stderr_text, file=sys.stderr)
                    elif (command_failure.get("stdout") or {}).get("text"):
                        print(command_failure["stdout"]["text"], file=sys.stderr)
                document_results.append(failure)
                print(f"{medicine.slug}/{language.value}: ERROR {failure['error']}", file=sys.stderr)

    xarelto_reference_tables = references_by_document.get(("xarelto", "en"), [])
    xarelto_case = cases_by_document.get(("xarelto", "en"))
    eliquis_en_reference_tables = references_by_document.get(("eliquis", "en"), [])
    eliquis_en_case = cases_by_document.get(("eliquis", "en"))
    negative_controls = run_negative_controls(
        xarelto_reference_tables,
        xarelto_case,
        eliquis_en_reference_tables,
        eliquis_en_case,
    )

    analyzed = [item for item in document_results if "comparison" in item]
    passed = [item for item in analyzed if item.get("acceptance_pass")]
    all_cases_pass = all(
        (item.get("oracle_case") or {}).get("pass", True)
        for item in analyzed
    )
    xarelto_result = next(
        (
            item.get("xarelto_independent_4_8_oracle")
            for item in analyzed
            if item.get("medicine") == "xarelto" and item.get("language") == "en"
        ),
        None,
    )
    summary = {
        "documents_requested": len(selected_medicines) * len(LANGUAGES),
        "documents_analyzed": len(analyzed),
        "documents_passed": len(passed),
        "pymupdf_pages": sum(item.get("page_count_pymupdf") or 0 for item in document_results),
        "native_pages": sum(item.get("page_count_native") or 0 for item in document_results),
        "first_copy_smpc_4_8_pages": sum(
            len((item.get("python_reference_first_copy_selection") or {}).get("owned_pages_4_8", []))
            for item in document_results
        ),
        "raw_reference_tables": sum(
            item.get("comparison", {}).get("reference_tables", 0) for item in document_results
        ),
        "reference_tables": sum(
            item.get("adjudicated_comparison", {}).get("reference_tables", 0)
            for item in document_results
        ),
        "native_tables": sum(
            item.get("adjudicated_comparison", {}).get("native_tables", 0)
            for item in document_results
        ),
        "reference_rows": sum(
            item.get("adjudicated_comparison", {}).get("reference_rows", 0)
            for item in document_results
        ),
        "native_rows": sum(
            item.get("adjudicated_comparison", {}).get("native_rows", 0)
            for item in document_results
        ),
        "reviewed_source_table_oracles_pass": all_cases_pass,
        "xarelto_en_independent_check_count": (
            xarelto_result.get("independent_check_count") if xarelto_result else None
        ),
        "xarelto_en_independent_oracle_pass": xarelto_result.get("pass") if xarelto_result else False,
        "negative_controls_pass": negative_controls["pass"],
    }
    overall_pass = (
        len(analyzed) == summary["documents_requested"]
        and len(passed) == summary["documents_requested"]
        and all_cases_pass
        and xarelto_result is not None
        and xarelto_result.get("pass") is True
        and negative_controls["pass"]
    )
    summary["status"] = "PASS" if overall_pass else "FAIL"

    versions = comparison.get_versions()
    metadata = cargo_metadata()
    archive_path = comparison.ASSET_DIR / "pdfium-linux-x64-8125.tgz"
    report = {
        "experiment": "Native Rust table detection and flattening versus MedLeaf's Python production table extraction",
        "attribution": comparison.SOURCE_ATTRIBUTION,
        "status": summary["status"],
        "manifest_sha256": manifest_hash,
        "sample_slugs": [medicine.slug for medicine in selected_medicines],
        "languages": list(LANGUAGES),
        "reproduction_command": shlex.join([
            sys.executable,
            "experiments/rust-pdf/run_native_tables.py",
            *(["--all"] if args.all else []),
            *(["--cached-only"] if getattr(args, "cached_only", False) else []),
            "--output", str(args.output),
            *(["--pdfium-library", args.pdfium_library] if args.pdfium_library else []),
        ]),
        "acceptance_thresholds": {
            "table_detection": {
                "precision_min": TABLE_DETECTION_MIN_PRECISION,
                "recall_min": TABLE_DETECTION_MIN_RECALL,
                "matching": f"one-to-one same-page table bbox IoU >= {TABLE_BBOX_IOU_MIN:.2f}",
            },
            "cell_content": {
                "precision_min": CELL_CONTENT_MIN_PRECISION,
                "recall_min": CELL_CONTENT_MIN_RECALL,
                "comparison": "NFKC+casefold whitespace-insensitive ordered character sequence, compared by ordered alignment over nonempty matched-table cells; punctuation including numeric separators is retained.",
            },
            "frequency_association": {
                "precision_min": FREQUENCY_MIN_PRECISION,
                "recall_min": FREQUENCY_MIN_RECALL,
                "comparison": "Ordered source-table frequency/reaction associations; reaction word order and clinically meaningful qualifiers are preserved.",
            },
            "column_frequency_association": {
                "precision_min": COLUMN_FREQUENCY_MIN_PRECISION,
                "recall_min": COLUMN_FREQUENCY_MIN_RECALL,
                "assignment_mismatches_allowed": 0,
                "applicability": "A positive source reaction/column/frequency association is required when the layout supports mapping; otherwise a positive row-frequency oracle must pass. Empty 0/0 comparisons never pass.",
                "key": "whitespace-insensitive ordered NFKC/casefold reaction text units + physical zero-based frequency/assignment column + normalized frequency label",
            },
            "exact_normalized_row_recall": "reported as a diagnostic only; not a pass gate",
            "existing_xarelto_en_oracle": "all eight independent assertions must pass",
        },
        "comparison_schema": {
            "candidate": "Rust JSON pages[*].tables[*].{bbox_pt,rows[*].cells[*],lines}; cell text/bold/bboxes and flattened lines are read directly from Rust output.",
            "reference_table_detection": "PyMuPDF page.find_tables() on Python-selected first-copy SmPC 4.8 pages, with per-document source-reviewed table adjudications and pinned-source fixtures where PyMuPDF has no reliable table.",
            "reference_cells": "Raw PyMuPDF Table.extract() cells are retained for the raw comparison; adjudicated scoring adds only manually transcribed pinned-source fixture cells and excludes individually documented false detections.",
            "reference_flattening": "The existing production flatten_table_rows() is applied to raw PyMuPDF reference cells only; source fixture rows are explicitly transcribed from the pinned PDF. Neither path uses native candidate cells.",
            "reference_frequency_associations": "When the source layout requires manual association adjudication, reaction/frequency pairs are transcribed from pinned source cells with per-association source provenance; no expected pair is derived from native output.",
            "native_flattening": "Native candidate flattened lines and rows are read directly from Rust output; Python does not rebuild them from native cells.",
            "table_identity": "One-to-one greedy matching of candidate/reference bbox IoU on the same page; geometry is comparison-only and does not construct candidate rows.",
            "row_identity": "NFKC+casefold, whitespace-collapsed exact line multiset; reported as a diagnostic.",
            "frequency_identity": "Whitespace-insensitive ordered source-table reaction text units keyed by normalized frequency; reaction order, punctuation, and clinical qualifiers are preserved.",
            "column_frequency_identity": "Whitespace-insensitive ordered reaction text units, physical frequency/assignment column, and frequency label are compared on matched tables. Applicability requires a positive reference mapping or a positive row-frequency alternate oracle.",
        },
        "selection": {
            "python_reference": "The existing Python production extract_lines/split_sections/first_copy_headings/_table_owner_pages implementation selects first-copy SmPC 4.8 owner pages. This is explicitly the reference selection, not a Rust page-selection result.",
            "native_candidates": "Only native tables on those Python-selected 4.8 pages are scored; native tables outside them are counted separately.",
        },
        "explicit_limits": [
            "The Python production parser selects section 4.8 pages and its first copy; this experiment does not measure native section-heading or repeated-copy selection.",
            "Raw PyMuPDF comparisons remain unchanged and are reported alongside source-adjudicated comparisons; source fixtures are manual pinned-PDF transcriptions with page/hash/provenance.",
            "Cell content uses whitespace-insensitive ordered character alignment (including punctuation); frequency and column gates preserve reaction order/qualifiers, require positive source evidence, and allow zero column assignments only when a positive alternate row oracle passes.",
            "Bbox IoU measures table detection agreement, not semantic equivalence; different but useful table boundaries can reduce detection scores.",
            "The reviewed bilingual oracles cover selected Eliquis Table 3 rows, not every adverse reaction, indication column, or table layout in the selected PDFs.",
        ],
        "tools": {
            **versions,
            "python_executable": sys.executable,
            "pdfium_library_path": str(library),
            "pdfium_library_sha256": library_hash,
            "pdfium_library_expected_sha256": comparison.PDFIUM_LIBRARY_SHA256,
            "pdfium_archive_url": comparison.PDFIUM_ARCHIVE_URL,
            "pdfium_archive_expected_sha256": comparison.PDFIUM_ARCHIVE_SHA256,
            "pdfium_archive_sha256": (
                comparison.sha256_file(archive_path) if archive_path.is_file() else None
            ),
            "cargo_build_command": build_command,
            "probe_binary_sha256": comparison.sha256_file(binary),
            "native_cases_sha256": comparison.sha256_file(CASES_PATH),
            "native_evaluator_sha256": comparison.sha256_file(Path(__file__).resolve()),
            **metadata,
        },
        "pdf_hashes": pdf_records,
        "negative_controls": negative_controls,
        "summary": summary,
        "documents": document_results,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate native Rust PDF tables against the pinned Python production reference."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--all", action="store_true", help="evaluate every medicine's EN/EL PDFs in the pinned manifest"
    )
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="evaluate all manifest documents from verified local assets only; missing PDFs remain errors and no downloads are attempted",
    )
    parser.add_argument(
        "--pdfium-library",
        help="optional already-downloaded pinned libpdfium.so; its pinned SHA-256 is verified",
    )
    args = parser.parse_args()

    try:
        report = run_evaluation(args)
    except Exception as error:
        report = {
            "experiment": "Native Rust table detection and flattening versus MedLeaf's Python production table extraction",
            "attribution": comparison.SOURCE_ATTRIBUTION,
            "status": "ERROR",
            "fatal_error": f"{type(error).__name__}: {error}",
            "manifest_sha256_expected": comparison.MANIFEST_SHA256,
            "acceptance_thresholds": {
                "table_detection_precision_min": TABLE_DETECTION_MIN_PRECISION,
                "table_detection_recall_min": TABLE_DETECTION_MIN_RECALL,
                "cell_content_precision_min": CELL_CONTENT_MIN_PRECISION,
                "cell_content_recall_min": CELL_CONTENT_MIN_RECALL,
                "frequency_precision_min": FREQUENCY_MIN_PRECISION,
                "frequency_recall_min": FREQUENCY_MIN_RECALL,
                "column_frequency_precision_min": COLUMN_FREQUENCY_MIN_PRECISION,
                "column_frequency_recall_min": COLUMN_FREQUENCY_MIN_RECALL,
                "column_frequency_applicability": (
                    "reference-supported grid or labelled layouts with at least one "
                    "reaction/column/frequency association"
                ),
            },
        }
        command_failure = command_failure_details(error)
        if command_failure is not None:
            report["command_failure"] = command_failure
        print(report["fatal_error"], file=sys.stderr)
        if command_failure is not None:
            stderr_text = (command_failure.get("stderr") or {}).get("text")
            if stderr_text:
                print(stderr_text, file=sys.stderr)
            elif (command_failure.get("stdout") or {}).get("text"):
                print(command_failure["stdout"]["text"], file=sys.stderr)
    try:
        write_result(args.output, report)
    except Exception as error:
        print(f"could not write result {args.output}: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    summary = report.get("summary", {"status": report.get("status")})
    print(json.dumps({"report": str(args.output), "summary": summary}, ensure_ascii=False))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
