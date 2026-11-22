from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = Path("/tmp/medleaf-pdf-assets")
CARGO_HOME = Path("/tmp/medleaf-pdf-cargo")
TARGET_DIR = Path("/tmp/medleaf-pdf-target")
MANIFEST_SHA256 = "281ac3a82c9d4cfa6ec44165ba5bce581d0151f8b0b55fdd52c94854065d9106"
PDFIUM_RELEASE = "158.0.8125.0"
PDFIUM_ARCHIVE_URL = (
    "https://github.com/bblanchon/pdfium-binaries/releases/download/"
    "chromium/8125/pdfium-linux-x64.tgz"
)
PDFIUM_ARCHIVE_SHA256 = "e1c3ce6e270cb5e0c20816b379ad9f2c1309972697c505f508de92bf5e7d6659"
PDFIUM_LIBRARY_SHA256 = "1c251efce460d67437a4c76d4be9033ae2f17bf5603f5bff1af10209b94ca723"
PDFIUM_RENDER_CRATE = "0.9.6"
SOURCE_ATTRIBUTION = (
    "Source: European Medicines Agency (EMA), EPAR product information. "
    "Extracted text and cell excerpts in this result remain attributed to EMA."
)
LINE_BASELINE_TOLERANCE_PT = 2.5
CELL_BOLD_THRESHOLD = 650
TEXT_RECALL_MIN = 0.98
HEADING_RECALL_MIN = 0.90
CELL_CHAR_RECALL_MIN = 0.95
FREQUENCY_ASSOCIATION_RECALL_MIN = 0.95
PAGE_SIZE_DELTA_PT = 0.5

sys.path.insert(0, str(ROOT / "src"))

import fitz  # noqa: E402 - the production parser imports the same PyMuPDF API
import pydantic  # noqa: E402
import yaml  # noqa: E402

from medleaf.ingest.chunk import (  # noqa: E402
    _cell_bold_flags,
    _page_bold_spans,
    flatten_table_rows,
    is_readable_frequency_row,
)
from medleaf.ingest.pdf import (  # noqa: E402
    SMPC_HEADING_RE,
    SMPC_NUMBERS,
    clean_line,
    detect_language,
    extract_lines,
    first_copy_headings,
    normalize_heading_line,
    split_sections,
)
from medleaf.ingest.pipeline import _table_owner_pages, parse_pdf  # noqa: E402
from medleaf.ingest.store import load_medicines  # noqa: E402
from medleaf.models import DocKind, Lang  # noqa: E402
from medleaf.sections import smpc_title  # noqa: E402


XARELTO_EN_ORACLE = {
    "Anaemia": "Common",
    "Thrombocytosis": "Uncommon",
    "Eye haemorrhage": "Common",
    "Hypotension": "Common",
}
XARELTO_SOC_HEADINGS = ("Eye disorders", "Nervous system disorders", "Vascular disorders")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_pinned(url: str, destination: Path, expected_sha256: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise ValueError(f"existing asset hash mismatch: {destination}: {actual}")
        return actual

    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "medleaf-rust-pdf-probe/0.1"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            actual = sha256_file(temporary)
            if actual != expected_sha256:
                temporary.unlink(missing_ok=True)
                raise ValueError(
                    f"pinned hash mismatch for {url}: expected {expected_sha256}, got {actual}"
                )
            temporary.replace(destination)
            return actual
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable retry loop")


def ensure_pdfium(library_arg: str | None) -> tuple[Path, str]:
    if library_arg:
        library = Path(library_arg).resolve()
        actual = sha256_file(library)
        if actual != PDFIUM_LIBRARY_SHA256:
            raise ValueError(f"PDFium library hash mismatch: expected {PDFIUM_LIBRARY_SHA256}, got {actual}")
        return library, actual

    archive = ASSET_DIR / "pdfium-linux-x64-8125.tgz"
    download_pinned(PDFIUM_ARCHIVE_URL, archive, PDFIUM_ARCHIVE_SHA256)
    library = ASSET_DIR / "pdfium-8125" / "libpdfium.so"
    if library.exists():
        actual = sha256_file(library)
        if actual != PDFIUM_LIBRARY_SHA256:
            raise ValueError(f"existing PDFium library hash mismatch: {library}: {actual}")
        return library, actual

    library.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.getmember("lib/libpdfium.so")
        if not member.isfile():
            raise ValueError("pinned PDFium archive does not contain a regular lib/libpdfium.so")
        source = bundle.extractfile(member)
        if source is None:
            raise ValueError("could not read lib/libpdfium.so from pinned PDFium archive")
        temporary = library.with_suffix(".so.part")
        with source, temporary.open("xb") as output:
            shutil.copyfileobj(source, output)
        actual = sha256_file(temporary)
        if actual != PDFIUM_LIBRARY_SHA256:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"extracted PDFium library hash mismatch: {actual}")
        temporary.replace(library)
    return library, actual


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def compact(text: str) -> str:
    return "".join(char for char in normalize(text) if char.isalnum() or char == ".")


def character_counter(text: str, *, greek_only: bool = False) -> collections.Counter[str]:
    counter: collections.Counter[str] = collections.Counter()
    for char in normalize(text):
        if char.isspace():
            continue
        is_greek = "GREEK" in unicodedata.name(char, "") and char.isalpha()
        if not greek_only or is_greek:
            counter[char] += 1
    return counter


def count_match(reference: collections.Counter[str], candidate: collections.Counter[str]) -> int:
    return sum((reference & candidate).values())


def rust_lines(page: dict[str, Any]) -> list[dict[str, Any]]:
    positioned = [
        glyph
        for glyph in page["chars"]
        if glyph["text"] and glyph["origin_pt"] is not None and glyph["text"] not in ("\n", "\r")
    ]
    positioned.sort(key=lambda glyph: (glyph["origin_pt"][1], glyph["origin_pt"][0], glyph["index"]))
    clusters: list[dict[str, Any]] = []
    for glyph in positioned:
        y = glyph["origin_pt"][1]
        if not clusters or abs(y - clusters[-1]["baseline_y"]) > LINE_BASELINE_TOLERANCE_PT:
            clusters.append({"baseline_y": y, "glyphs": [glyph]})
        else:
            cluster = clusters[-1]
            cluster["glyphs"].append(glyph)
            cluster["baseline_y"] = sum(
                item["origin_pt"][1] for item in cluster["glyphs"]
            ) / len(cluster["glyphs"])

    lines: list[dict[str, Any]] = []
    for cluster in clusters:
        glyphs = sorted(cluster["glyphs"], key=lambda glyph: (glyph["origin_pt"][0], glyph["index"]))
        parts: list[str] = []
        previous_right: float | None = None
        boxes = [glyph["bbox_pt"] for glyph in glyphs if glyph["bbox_pt"] is not None]
        for glyph in glyphs:
            text = glyph["text"] or ""
            box = glyph["bbox_pt"]
            x = box[0] if box is not None else glyph["origin_pt"][0]
            right = box[2] if box is not None else glyph["origin_pt"][0]
            if text.isspace():
                if parts and parts[-1] != " ":
                    parts.append(" ")
                previous_right = max(previous_right or right, right)
                continue
            if previous_right is not None and x - previous_right > 1.4 and parts and parts[-1] != " ":
                parts.append(" ")
            parts.append(text)
            previous_right = max(previous_right or right, right)

        bbox = None
        if boxes:
            bbox = [
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            ]
        lines.append(
            {
                "text": re.sub(r"\s+", " ", "".join(parts)).strip(),
                "baseline_y_pt": cluster["baseline_y"],
                "x_pt": min(glyph["origin_pt"][0] for glyph in glyphs),
                "bbox_pt": bbox,
            }
        )
    return lines


def valid_smpc_headings(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        text = normalize_heading_line(clean_line(line["text"]))
        match = SMPC_HEADING_RE.match(text)
        if not match:
            continue
        number = match.group("num")
        title = match.group("title").strip()
        if number in SMPC_NUMBERS and smpc_title(number, title):
            headings.append(
                {"key": number, "title": title, "page": line["page"], "index": index}
            )
    return headings


def page_text_metrics(
    reference_lines: list[Any], rust_document: dict[str, Any], pymupdf_page_sizes: list[tuple[float, float]]
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    reference_by_page: dict[int, list[str]] = collections.defaultdict(list)
    for line in reference_lines:
        reference_by_page[line.page].append(line.text)

    rust_page_by_number = {page["page"]: page for page in rust_document["pages"]}
    rust_lines_by_page: dict[int, list[dict[str, Any]]] = {}
    reference_total: collections.Counter[str] = collections.Counter()
    rust_total: collections.Counter[str] = collections.Counter()
    reference_greek: collections.Counter[str] = collections.Counter()
    rust_greek: collections.Counter[str] = collections.Counter()
    matched = 0
    greek_matched = 0
    page_covered = 0
    page_recall_values: list[float] = []
    line_reference_total = 0
    line_matches = 0
    max_width_delta = 0.0
    max_height_delta = 0.0
    unmapped = 0
    replacement_chars = 0

    for page_number in range(1, len(rust_document["pages"]) + 1):
        ref_text = "\n".join(reference_by_page.get(page_number, []))
        rust_page = rust_page_by_number[page_number]
        rust_text = "".join(glyph["text"] or "" for glyph in rust_page["chars"])
        ref_chars = character_counter(ref_text)
        rust_chars = character_counter(rust_text)
        ref_greek_chars = character_counter(ref_text, greek_only=True)
        rust_greek_chars = character_counter(rust_text, greek_only=True)
        reference_total.update(ref_chars)
        rust_total.update(rust_chars)
        reference_greek.update(ref_greek_chars)
        rust_greek.update(rust_greek_chars)
        page_matched = count_match(ref_chars, rust_chars)
        page_ref_count = sum(ref_chars.values())
        matched += page_matched
        greek_matched += count_match(ref_greek_chars, rust_greek_chars)
        page_recall = page_matched / page_ref_count if page_ref_count else 1.0
        if page_ref_count:
            page_recall_values.append(page_recall)
            if page_recall >= 0.90:
                page_covered += 1

        rust_page_lines = rust_lines(rust_page)
        rust_lines_by_page[page_number] = [dict(line, page=page_number) for line in rust_page_lines]
        ref_line_counter = collections.Counter(
            " ".join(clean_line(text).split()).casefold()
            for text in reference_by_page.get(page_number, [])
            if clean_line(text)
        )
        rust_line_counter = collections.Counter(
            " ".join(clean_line(line["text"]).split()).casefold()
            for line in rust_page_lines
            if clean_line(line["text"])
        )
        line_reference_total += sum(ref_line_counter.values())
        line_matches += count_match(ref_line_counter, rust_line_counter)

        if page_number <= len(pymupdf_page_sizes):
            ref_width, ref_height = pymupdf_page_sizes[page_number - 1]
            max_width_delta = max(max_width_delta, abs(ref_width - rust_page["width_pt"]))
            max_height_delta = max(max_height_delta, abs(ref_height - rust_page["height_pt"]))

        for glyph in rust_page["chars"]:
            if glyph["text"] is None:
                unmapped += 1
            if glyph["text"] == "\ufffd":
                replacement_chars += 1

    ref_total = sum(reference_total.values())
    rust_total_count = sum(rust_total.values())
    ref_greek_total = sum(reference_greek.values())
    rust_greek_total = sum(rust_greek.values())
    metrics = {
        "pymupdf_nonspace_chars": ref_total,
        "pdfium_nonspace_chars": rust_total_count,
        "normalized_char_multiset_matched": matched,
        "normalized_char_multiset_recall": round(matched / ref_total, 6) if ref_total else 1.0,
        "normalized_char_multiset_precision": round(matched / rust_total_count, 6) if rust_total_count else 1.0,
        "reference_pages_with_text": len(page_recall_values),
        "pages_at_least_90pct_char_recall": page_covered,
        "minimum_nonempty_page_char_recall": round(min(page_recall_values), 6) if page_recall_values else 1.0,
        "greek_reference_chars": ref_greek_total,
        "greek_pdfium_chars": rust_greek_total,
        "greek_char_multiset_recall": round(greek_matched / ref_greek_total, 6) if ref_greek_total else None,
        "pdfium_unmapped_unicode_chars": unmapped,
        "pdfium_replacement_chars": replacement_chars,
        "pymupdf_visual_lines": line_reference_total,
        "pdfium_visual_lines": sum(len(lines) for lines in rust_lines_by_page.values()),
        "exact_normalized_line_matches": line_matches,
        "exact_normalized_line_recall": round(line_matches / line_reference_total, 6) if line_reference_total else 1.0,
        "max_page_width_delta_pt": round(max_width_delta, 4),
        "max_page_height_delta_pt": round(max_height_delta, 4),
    }
    return metrics, rust_lines_by_page


def heading_metrics(
    raw_lines: list[Any], rust_lines_by_page: dict[int, list[dict[str, Any]]]
) -> dict[str, Any]:
    accepted, copy_end = first_copy_headings(raw_lines, doc=DocKind.SMPC)
    candidates_python = valid_smpc_headings(
        [{"text": line.text, "page": line.page} for line in raw_lines]
    )
    all_rust_lines = [line for page_lines in rust_lines_by_page.values() for line in page_lines]
    candidates_rust = valid_smpc_headings(all_rust_lines)
    exact_line_hits = 0
    same_page_text_hits = 0
    for hit in accepted:
        reference_line = clean_line(raw_lines[hit.index].text)
        canonical = " ".join(normalize(reference_line).split())
        page_lines = rust_lines_by_page.get(raw_lines[hit.index].page, [])
        if any(" ".join(normalize(clean_line(line["text"])).split()) == canonical for line in page_lines):
            exact_line_hits += 1
        rust_page_compact = compact(" ".join(line["text"] for line in page_lines))
        if compact(reference_line) in rust_page_compact:
            same_page_text_hits += 1

    def copy_starts(headings: list[dict[str, Any]]) -> int:
        return sum(
            1
            for index, heading in enumerate(headings[:-2])
            if heading["key"] == "1"
            and [item["key"] for item in headings[index + 1 : index + 3]] == ["2", "3"]
        )

    counts_python = collections.Counter(item["key"] for item in candidates_python)
    counts_rust = collections.Counter(item["key"] for item in candidates_rust)
    denominator = len(accepted)
    return {
        "first_copy_sections_selected_by_python": len(accepted),
        "first_copy_accepted_heading_count": denominator,
        "pdfium_exact_line_heading_matches": exact_line_hits,
        "pdfium_same_page_heading_text_matches": same_page_text_hits,
        "pdfium_same_page_heading_text_recall": round(same_page_text_hits / denominator, 6) if denominator else 1.0,
        "python_valid_heading_candidates": len(candidates_python),
        "pdfium_valid_heading_candidates": len(candidates_rust),
        "python_4_8_heading_occurrences": counts_python.get("4.8", 0),
        "pdfium_4_8_heading_occurrences": counts_rust.get("4.8", 0),
        "python_1_to_2_to_3_copy_start_candidates": copy_starts(candidates_python),
        "pdfium_1_to_2_to_3_copy_start_candidates": copy_starts(candidates_rust),
        "python_first_copy_end_page": raw_lines[copy_end - 1].page if copy_end and copy_end <= len(raw_lines) else None,
    }


def glyph_center(glyph: dict[str, Any]) -> tuple[float, float] | None:
    box = glyph["bbox_pt"]
    if box is not None and all(math.isfinite(float(value)) for value in box):
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    origin = glyph["origin_pt"]
    if origin is not None and all(math.isfinite(float(value)) for value in origin):
        return origin[0], origin[1]
    return None


def glyphs_in_cell(
    glyphs: list[dict[str, Any]], rect: tuple[float, float, float, float] | None
) -> list[dict[str, Any]]:
    if rect is None:
        return []
    left, top, right, bottom = rect
    selected = []
    for glyph in glyphs:
        point = glyph_center(glyph)
        if point is not None and left <= point[0] <= right and top <= point[1] <= bottom:
            selected.append(glyph)
    return selected


def spatial_text(glyphs: list[dict[str, Any]]) -> str:
    positioned = [
        glyph for glyph in glyphs
        if glyph["text"] and glyph["text"] not in ("\n", "\r") and glyph["origin_pt"] is not None
    ]
    positioned.sort(key=lambda glyph: (glyph["origin_pt"][1], glyph["origin_pt"][0], glyph["index"]))
    lines: list[list[dict[str, Any]]] = []
    baselines: list[float] = []
    for glyph in positioned:
        y = glyph["origin_pt"][1]
        if not lines or abs(y - baselines[-1]) > LINE_BASELINE_TOLERANCE_PT:
            lines.append([glyph])
            baselines.append(y)
        else:
            lines[-1].append(glyph)
            baselines[-1] = sum(item["origin_pt"][1] for item in lines[-1]) / len(lines[-1])

    rendered: list[str] = []
    for line in lines:
        parts: list[str] = []
        previous_right: float | None = None
        for glyph in sorted(line, key=lambda item: (item["origin_pt"][0], item["index"])):
            text = glyph["text"] or ""
            box = glyph["bbox_pt"]
            x = box[0] if box is not None else glyph["origin_pt"][0]
            right = box[2] if box is not None else glyph["origin_pt"][0]
            if text.isspace():
                if parts and parts[-1] != " ":
                    parts.append(" ")
                previous_right = max(previous_right or right, right)
            else:
                if previous_right is not None and x - previous_right > 1.4 and parts and parts[-1] != " ":
                    parts.append(" ")
                parts.append(text)
                previous_right = max(previous_right or right, right)
        rendered.append(re.sub(r"\s+", " ", "".join(parts)).strip())
    return " ".join(line for line in rendered if line)


def pdfium_cell_bold(glyphs: list[dict[str, Any]]) -> bool | None:
    weights = [glyph["font_weight"] for glyph in glyphs if glyph["font_weight"] is not None]
    if not weights:
        return None
    bold_votes = sum(weight >= CELL_BOLD_THRESHOLD for weight in weights)
    ratio = bold_votes / len(weights)
    if ratio >= 0.60:
        return True
    if ratio <= 0.40:
        return False
    return None


def rect_values(cell: Any) -> tuple[float, float, float, float] | None:
    if cell is None:
        return None
    values = tuple(float(value) for value in cell)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    return values  # PyMuPDF coordinates are top-left-origin points.


def frequency_association_key(line: str) -> tuple[tuple[tuple[str, int], ...], str] | None:
    if not is_readable_frequency_row(line) or " - " not in line:
        return None
    first_cell, _, remaining_cells = line.partition(" - ")
    frequency_match = re.search(
        r"(?:frequency|συχνότητα)\s*:\s*(.*?)(?=;\s*(?:adverse reaction|ανεπιθύμητη ενέργεια)\s*:|$)",
        remaining_cells,
        re.IGNORECASE,
    )
    reaction_match = re.search(
        r"(?:adverse reaction|ανεπιθύμητη ενέργεια)\s*:\s*(.*)$",
        remaining_cells,
        re.IGNORECASE,
    )
    if frequency_match and reaction_match:
        reaction = reaction_match.group(1)
        frequency = compact(frequency_match.group(1))
    else:
        reaction = first_cell
        frequency = compact(remaining_cells)
    reaction_chars = collections.Counter(char for char in normalize(reaction) if char.isalnum())
    if not reaction_chars or not frequency:
        return None
    return tuple(sorted(reaction_chars.items())), frequency


def compare_tables(
    pdf_path: Path,
    owned_pages: set[int],
    rust_document: dict[str, Any],
    *,
    keep_cells: bool,
) -> dict[str, Any]:
    rust_pages = {page["page"]: page for page in rust_document["pages"]}
    table_reports: list[dict[str, Any]] = []
    ref_cells_total = 0
    exact_cells = 0
    char_reference_total = 0
    char_matched_total = 0
    bold_comparable = 0
    bold_matched = 0
    ref_flat: list[str] = []
    rust_flat: list[str] = []

    with fitz.open(pdf_path) as document:
        for page_number in sorted(owned_pages):
            if not 1 <= page_number <= document.page_count:
                continue
            page = document[page_number - 1]
            page_tables = page.find_tables().tables
            if not page_tables:
                continue
            spans = _page_bold_spans(page)
            rust_glyphs = rust_pages[page_number]["chars"]
            for table_index, table in enumerate(page_tables):
                reference_rows = [[cell or "" for cell in row] for row in table.extract()]
                row_objects = getattr(table, "rows", None)
                ref_bolds = _cell_bold_flags(row_objects, spans) if row_objects is not None else None
                geometry = [
                    [rect_values(cell) for cell in row.cells]
                    for row in row_objects
                ] if row_objects is not None else []
                rust_rows: list[list[str]] = []
                rust_bolds: list[list[bool | None]] = []
                row_count = max(len(reference_rows), len(geometry))
                for row_index in range(row_count):
                    ref_row = reference_rows[row_index] if row_index < len(reference_rows) else []
                    rect_row = geometry[row_index] if row_index < len(geometry) else []
                    column_count = max(len(ref_row), len(rect_row))
                    rust_row: list[str] = []
                    rust_bold_row: list[bool | None] = []
                    for column_index in range(column_count):
                        rect = rect_row[column_index] if column_index < len(rect_row) else None
                        selected = glyphs_in_cell(rust_glyphs, rect)
                        rust_text = spatial_text(selected)
                        reference_text = ref_row[column_index] if column_index < len(ref_row) else ""
                        rust_row.append(rust_text)
                        rust_bold_row.append(pdfium_cell_bold(selected))
                        if reference_text.strip():
                            ref_cells_total += 1
                            ref_counter = collections.Counter(
                                char for char in normalize(reference_text) if char.isalnum()
                            )
                            rust_counter = collections.Counter(
                                char for char in normalize(rust_text) if char.isalnum()
                            )
                            char_reference_total += sum(ref_counter.values())
                            char_matched_total += count_match(ref_counter, rust_counter)
                            if " ".join(normalize(reference_text).split()) == " ".join(normalize(rust_text).split()):
                                exact_cells += 1
                        if (
                            ref_bolds is not None
                            and row_index < len(ref_bolds)
                            and column_index < len(ref_bolds[row_index])
                            and ref_bolds[row_index][column_index] is not None
                            and rust_bold_row[-1] is not None
                        ):
                            bold_comparable += 1
                            bold_matched += ref_bolds[row_index][column_index] == rust_bold_row[-1]
                    rust_rows.append(rust_row)
                    rust_bolds.append(rust_bold_row)

                reference_flattened = flatten_table_rows(reference_rows, ref_bolds)
                rust_flattened = flatten_table_rows(rust_rows, rust_bolds)
                ref_flat.extend(reference_flattened)
                rust_flat.extend(rust_flattened)
                table_report: dict[str, Any] = {
                    "page": page_number,
                    "table_index": table_index,
                    "reference_rows": len(reference_rows),
                    "columns": max((len(row) for row in reference_rows), default=0),
                    "nonempty_reference_cells": sum(bool(cell.strip()) for row in reference_rows for cell in row),
                    "pdfium_nonempty_cells": sum(bool(cell.strip()) for row in rust_rows for cell in row),
                    "exact_nonempty_cell_text_matches": sum(
                        1
                        for ref_row, rust_row in zip(reference_rows, rust_rows)
                        for ref_cell, rust_cell in zip(ref_row, rust_row)
                        if ref_cell.strip()
                        and " ".join(normalize(ref_cell).split()) == " ".join(normalize(rust_cell).split())
                    ),
                    "reference_flattened_frequency_rows": sum(is_readable_frequency_row(line) for line in reference_flattened),
                    "pdfium_geometry_flattened_frequency_rows": sum(is_readable_frequency_row(line) for line in rust_flattened),
                }
                if keep_cells:
                    table_report["source_excerpt"] = {
                        "attribution": SOURCE_ATTRIBUTION,
                        "reference_cells": reference_rows,
                        "pdfium_geometry_cells": rust_rows,
                        "reference_flattened_rows": reference_flattened,
                        "pdfium_geometry_flattened_rows": rust_flattened,
                    }
                table_reports.append(table_report)

    ref_flat_frequency = collections.Counter(
        " ".join(normalize(line).split()) for line in ref_flat if is_readable_frequency_row(line)
    )
    rust_flat_frequency = collections.Counter(
        " ".join(normalize(line).split()) for line in rust_flat if is_readable_frequency_row(line)
    )
    frequency_match = count_match(ref_flat_frequency, rust_flat_frequency)

    ref_associations: collections.Counter[Any] = collections.Counter()
    rust_associations: collections.Counter[Any] = collections.Counter()
    for line in ref_flat:
        key = frequency_association_key(line)
        if key is not None:
            ref_associations[key] += 1
    for line in rust_flat:
        key = frequency_association_key(line)
        if key is not None:
            rust_associations[key] += 1
    association_match = count_match(ref_associations, rust_associations)

    return {
        "section": "4.8",
        "page_ownership_source": "existing MedLeaf Python parser's _table_owner_pages() first-copy section span",
        "cell_boundary_source": "PyMuPDF TableFinder cell rectangles; PDFium glyph bbox centers are assigned inside them",
        "cell_geometry_expansion_pt": 0.0,
        "tables_found": len(table_reports),
        "nonempty_reference_cells": ref_cells_total,
        "exact_nonempty_cell_text_matches": exact_cells,
        "exact_nonempty_cell_match_rate": round(exact_cells / ref_cells_total, 6) if ref_cells_total else None,
        "reference_cell_alphanumeric_chars": char_reference_total,
        "pdfium_cell_alphanumeric_char_recall": round(char_matched_total / char_reference_total, 6) if char_reference_total else None,
        "bold_cells_comparable": bold_comparable,
        "bold_cells_equal_to_pymupdf": bold_matched,
        "pdfium_font_weight_bold_threshold": CELL_BOLD_THRESHOLD,
        "bold_cell_agreement": round(bold_matched / bold_comparable, 6) if bold_comparable else None,
        "reference_flattened_frequency_rows": sum(is_readable_frequency_row(line) for line in ref_flat),
        "pdfium_geometry_flattened_frequency_rows": sum(is_readable_frequency_row(line) for line in rust_flat),
        "pdfium_exact_frequency_rows_matching_reference": frequency_match,
        "exact_flattened_frequency_row_recall": round(frequency_match / sum(ref_flat_frequency.values()), 6) if ref_flat_frequency else None,
        "reference_frequency_associations": sum(ref_associations.values()),
        "pdfium_frequency_associations": sum(rust_associations.values()),
        "pdfium_frequency_associations_matching_reference": association_match,
        "frequency_association_recall": round(association_match / sum(ref_associations.values()), 6) if ref_associations else None,
        "frequency_association_key": "reaction-cell alphanumeric character multiset + frequency label; punctuation and order ignored",
        "tables": table_reports,
        "reference_flattened_rows": ref_flat,
        "pdfium_geometry_flattened_rows": rust_flat,
    }


def xarelto_oracle(lines: list[str]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    matched_rows: dict[str, list[str]] = {}
    for reaction, frequency in XARELTO_EN_ORACLE.items():
        found = [
            line for line in lines
            if line.casefold().startswith(reaction.casefold()) and frequency.casefold() in line.casefold()
        ]
        checks[f"{reaction} -> {frequency}"] = bool(found)
        matched_rows[reaction] = found[:3]
    glued = [
        line for line in lines
        if line.casefold().startswith("anaemia") and "thrombocytosis" in line.casefold()
    ]
    checks["Anaemia not glued to Thrombocytosis"] = not glued
    for heading in XARELTO_SOC_HEADINGS:
        bad = [
            line for line in lines
            if line.casefold().startswith(heading.casefold())
            and any(f" - {frequency}" in line.casefold() for frequency in ("common", "rare"))
        ]
        checks[f"SOC {heading} stays unbanded"] = not bad
        if bad:
            matched_rows[heading] = bad[:3]
    return {
        "oracle_source": "tests/test_m1_acceptance.py:208-264 (independent real-PDF acceptance assertions)",
        "checks": checks,
        "pass": all(checks.values()),
        "matched_rows": matched_rows,
        "attribution": SOURCE_ATTRIBUTION,
    }


def get_versions() -> dict[str, Any]:
    def version(command: list[str]) -> str | None:
        executable = shutil.which(command[0])
        if executable is None and command[0] == "cargo":
            executable = str(Path.home() / ".cargo" / "bin" / "cargo")
        if executable is None and command[0] == "rustc":
            executable = str(Path.home() / ".cargo" / "bin" / "rustc")
        if executable is None or not Path(executable).exists():
            return None
        try:
            return subprocess.run([executable, *command[1:]], check=True, capture_output=True, text=True).stdout.strip()
        except Exception:
            return None

    return {
        "python": platform.python_version(),
        "pymupdf": fitz.VersionBind,
        "pydantic": pydantic.__version__,
        "pyyaml": yaml.__version__,
        "rustc": version(["rustc", "--version"]),
        "cargo": version(["cargo", "--version"]),
        "uv": version(["uv", "--version"]),
        "pdfium_release": PDFIUM_RELEASE,
        "pdfium_render_crate": PDFIUM_RENDER_CRATE,
    }


def ensure_cargo_binary() -> Path:
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
    if not Path(cargo).exists():
        raise RuntimeError("cargo not found; add ~/.cargo/bin to PATH")
    return Path(cargo)


def run_probe(binary: Path, pdf_path: Path, library: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["PDFIUM_LIBRARY_PATH"] = str(library)
    process = subprocess.run(
        [str(binary), str(pdf_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(process.stdout)


def analyze_document(medicine: Any, lang: Lang, pdf_path: Path, digest: str, rust: dict[str, Any]) -> dict[str, Any]:
    document_ref = medicine.documents[lang]
    raw_lines = extract_lines(str(pdf_path))
    raw_smpc = split_sections(raw_lines, doc=DocKind.SMPC)
    accepted, _ = first_copy_headings(raw_lines, doc=DocKind.SMPC)
    parsed = parse_pdf(
        medicine=medicine,
        lang=lang,
        pdf_path=pdf_path,
        corpus_version=MANIFEST_SHA256,
        max_tokens=400,
    )
    section_48 = [section for section in parsed.by_doc(DocKind.SMPC) if section.section == "4.8"]
    section_48_text = "\n".join(section.text for section in section_48)
    section_48_lines = [line.strip() for line in section_48_text.splitlines() if line.strip()]
    section_frequency_rows = [line for line in section_48_lines if is_readable_frequency_row(line)]
    owner_pages = {
        page for page, owner in _table_owner_pages(raw_lines, raw_smpc).items() if owner == "4.8"
    }

    with fitz.open(pdf_path) as source:
        pymupdf_pages = source.page_count
        page_sizes = [(page.rect.width, page.rect.height) for page in source]
    extraction, rust_lines_by_page = page_text_metrics(raw_lines, rust, page_sizes)
    headings = heading_metrics(raw_lines, rust_lines_by_page)
    table_comparison = compare_tables(
        pdf_path,
        owner_pages,
        rust,
        keep_cells=(medicine.slug == "xarelto" and lang is Lang.EN),
    )

    detected_lang = detect_language("\n".join(line.text for line in raw_lines if line.page <= 6))
    page_count_match = pymupdf_pages == len(rust["pages"])
    page_coverage_pass = (
        extraction["pages_at_least_90pct_char_recall"]
        == extraction["reference_pages_with_text"]
    )
    page_geometry_pass = (
        extraction["max_page_width_delta_pt"] <= PAGE_SIZE_DELTA_PT
        and extraction["max_page_height_delta_pt"] <= PAGE_SIZE_DELTA_PT
    )
    language_match = detected_lang is lang
    text_pass = (
        extraction["normalized_char_multiset_recall"] >= TEXT_RECALL_MIN
        and page_count_match
        and page_coverage_pass
        and page_geometry_pass
        and language_match
    )
    greek_recall = extraction["greek_char_multiset_recall"]
    if lang is Lang.EL:
        text_pass = text_pass and greek_recall is not None and greek_recall >= TEXT_RECALL_MIN
    heading_pass = headings["pdfium_same_page_heading_text_recall"] >= HEADING_RECALL_MIN
    cell_recall = table_comparison["pdfium_cell_alphanumeric_char_recall"]
    association_recall = table_comparison["frequency_association_recall"]
    cell_pass = (
        table_comparison["tables_found"] > 0
        and cell_recall is not None
        and cell_recall >= CELL_CHAR_RECALL_MIN
    )
    association_pass = (
        table_comparison["tables_found"] > 0
        and association_recall is not None
        and association_recall >= FREQUENCY_ASSOCIATION_RECALL_MIN
    )

    result: dict[str, Any] = {
        "medicine": medicine.slug,
        "language": lang.value,
        "source_url": str(document_ref.url),
        "source_attribution": SOURCE_ATTRIBUTION,
        "manifest_sha256": document_ref.sha256,
        "pdf_sha256": digest,
        "page_count_pymupdf": pymupdf_pages,
        "page_count_pdfium": len(rust["pages"]),
        "requested_language": lang.value,
        "detected_language_by_existing_pipeline": detected_lang.value,
        "language_match": detected_lang is lang,
        "pymupdf_first_copy_section_keys": [key for key, _, _ in raw_smpc],
        "pymupdf_first_copy_accepted_headings": len(accepted),
        "extraction": extraction,
        "layout_and_headings": headings,
        "smpc_4_8": {
            "python_parsed_chunks": len(section_48),
            "python_parsed_readable_frequency_rows": len(section_frequency_rows),
            "pdfium_owned_table_pages": sorted(owner_pages),
            "table_cell_geometry_comparison": table_comparison,
        },
        "criteria": {
            "pymupdf_pdfium_page_count_match": page_count_match,
            "all_reference_text_pages_at_least_90pct_char_recall": page_coverage_pass,
            "page_dimensions_within_0_5_pt": page_geometry_pass,
            "language_match": language_match,
            "text_character_multiset_recall_at_least_0_98": extraction["normalized_char_multiset_recall"] >= TEXT_RECALL_MIN,
            "text_extraction_pass": text_pass,
            "first_copy_heading_text_recall_at_least_0_90": heading_pass,
            "table_cell_char_recall_at_least_0_95_when_table_exists": cell_pass if owner_pages else None,
            "frequency_association_recall_at_least_0_95_when_table_exists": association_pass if owner_pages else None,
        },
    }
    if medicine.slug == "xarelto" and lang is Lang.EN:
        py_oracle = xarelto_oracle(section_48_lines)
        pdfium_oracle = xarelto_oracle(table_comparison["pdfium_geometry_flattened_rows"])
        result["xarelto_independent_4_8_oracle"] = {
            "pymupdf_production_parser": py_oracle,
            "pdfium_glyphs_in_pymupdf_cell_geometry": pdfium_oracle,
            "pass": py_oracle["pass"] and pdfium_oracle["pass"],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PDFium glyph extraction with MedLeaf's real PyMuPDF ingest pipeline.")
    parser.add_argument("--slugs", default="xarelto,eliquis,humira", help="comma-separated medicines; each selected medicine uses EN and EL")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/rust-pdf/results.json")
    parser.add_argument("--pdfium-library", help="optional already-downloaded pinned libpdfium.so")
    args = parser.parse_args()

    manifest = ROOT / "data/medicines.yaml"
    manifest_hash = sha256_file(manifest)
    if manifest_hash != MANIFEST_SHA256:
        raise ValueError(f"manifest changed: expected {MANIFEST_SHA256}, got {manifest_hash}; refusing to repin")
    medicines = load_medicines(manifest)
    by_slug = {medicine.slug: medicine for medicine in medicines}
    slugs = [slug.strip() for slug in args.slugs.split(",") if slug.strip()]
    if not slugs or len(set(slugs)) != len(slugs):
        raise ValueError("--slugs must contain one or more distinct manifest slugs")
    unknown = [slug for slug in slugs if slug not in by_slug]
    if unknown:
        raise ValueError(f"unknown manifest medicine slugs: {unknown}")

    library, library_hash = ensure_pdfium(args.pdfium_library)
    cargo = ensure_cargo_binary()
    CARGO_HOME.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    cargo_env = os.environ.copy()
    cargo_env["CARGO_HOME"] = str(CARGO_HOME)
    cargo_env["CARGO_TARGET_DIR"] = str(TARGET_DIR)
    cargo_env["CARGO_BUILD_JOBS"] = "1"
    cargo_env["PDFIUM_LIBRARY_PATH"] = str(library)
    cargo_env["PATH"] = f"{cargo.parent}:{cargo_env.get('PATH', '')}"
    cargo_args = [str(cargo), "build", "--release"]
    if (ROOT / "experiments/rust-pdf/Cargo.lock").exists():
        cargo_args.append("--locked")
    cargo_args.extend(["--manifest-path", str(ROOT / "experiments/rust-pdf/Cargo.toml")])
    subprocess.run(cargo_args, cwd=ROOT, env=cargo_env, check=True)
    binary = TARGET_DIR / "release/medleaf-ema-pdfium-probe"
    if not binary.exists():
        raise RuntimeError(f"cargo build completed without the expected probe: {binary}")

    results: list[dict[str, Any]] = []
    for slug in slugs:
        medicine = by_slug[slug]
        for lang in (Lang.EN, Lang.EL):
            document_ref = medicine.documents[lang]
            if not document_ref.sha256:
                raise ValueError(f"manifest has no pinned SHA-256 for {slug}/{lang.value}")
            pdf_path = ASSET_DIR / f"{slug}_{lang.value}.pdf"
            digest = download_pinned(str(document_ref.url), pdf_path, document_ref.sha256)
            with pdf_path.open("rb") as handle:
                if handle.read(5) != b"%PDF-":
                    raise ValueError(f"download is not a PDF: {pdf_path}")
            rust = run_probe(binary, pdf_path, library)
            results.append(analyze_document(medicine, lang, pdf_path, digest, rust))
            print(
                f"{slug}/{lang.value}: pages={results[-1]['page_count_pdfium']} "
                f"char_recall={results[-1]['extraction']['normalized_char_multiset_recall']:.4f} "
                f"heading_recall={results[-1]['layout_and_headings']['pdfium_same_page_heading_text_recall']:.4f} "
                f"4.8_tables={results[-1]['smpc_4_8']['table_cell_geometry_comparison']['tables_found']}"
            )

    doc_count = len(results)
    page_count = sum(item["page_count_pdfium"] for item in results)
    text_pass_count = sum(item["criteria"]["text_extraction_pass"] for item in results)
    heading_pass_count = sum(item["criteria"]["first_copy_heading_text_recall_at_least_0_90"] for item in results)
    table_cells = [
        item["criteria"]["table_cell_char_recall_at_least_0_95_when_table_exists"]
        for item in results
        if item["criteria"]["table_cell_char_recall_at_least_0_95_when_table_exists"] is not None
    ]
    table_associations = [
        item["criteria"]["frequency_association_recall_at_least_0_95_when_table_exists"]
        for item in results
        if item["criteria"]["frequency_association_recall_at_least_0_95_when_table_exists"] is not None
    ]
    xarelto = next((item for item in results if item["medicine"] == "xarelto" and item["language"] == "en"), None)
    status = (
        "PASS"
        if text_pass_count == doc_count
        and heading_pass_count == doc_count
        and all(table_cells)
        and all(table_associations)
        and xarelto
        and xarelto["xarelto_independent_4_8_oracle"]["pass"]
        else "FAIL"
    )
    summary = {
        "documents_processed": doc_count,
        "pages_processed": page_count,
        "text_extraction_pass_documents": text_pass_count,
        "heading_coverage_pass_documents": heading_pass_count,
        "table_cell_pass_documents_with_detected_4_8_tables": sum(table_cells),
        "table_cell_documents_with_detected_4_8_tables": len(table_cells),
        "frequency_association_pass_documents_with_detected_4_8_tables": sum(table_associations),
        "frequency_association_documents_with_detected_4_8_tables": len(table_associations),
        "xarelto_en_independent_oracle_pass": (
            xarelto["xarelto_independent_4_8_oracle"]["pass"] if xarelto is not None else None
        ),
        "status": status,
    }
    report = {
        "experiment": "Rust PDFium extraction feasibility versus the existing Python PyMuPDF pipeline",
        "attribution": SOURCE_ATTRIBUTION,
        "scope_warning": (
            "Extraction/layout feasibility only, not a ported-parser parity claim. The Rust probe returns Unicode glyphs, "
            "positions, bounding boxes and font weights; it does not split SmPC sections, recognize first copies, detect "
            "tables or associate frequency labels. The table-cell diagnostic deliberately borrows PyMuPDF TableFinder "
            "cell rectangles and the existing Python flatten_table_rows() function, so it does not establish a native Rust table parser."
        ),
        "manifest_sha256": manifest_hash,
        "sample_slugs": slugs,
        "tools": {
            **get_versions(),
            "pdfium_archive_url": PDFIUM_ARCHIVE_URL,
            "pdfium_archive_sha256": PDFIUM_ARCHIVE_SHA256,
            "pdfium_library_sha256": library_hash,
        },
        "tolerances_and_pass_criteria": {
            "text_comparison": "per-page Unicode NFKC + casefold character multiset, whitespace ignored; recall >= 0.98",
            "greek_comparison": "same character multiset restricted to Greek letters; paired EN/EL docs and Greek recall >= 0.98",
            "page_comparison": "exact PyMuPDF/PDFium page-count match, all nonempty reference pages >= 0.90 character recall, and page-size deltas <= 0.5 pt",
            "layout_lines": f"PDFium glyph origins grouped on baselines within {LINE_BASELINE_TOLERANCE_PT} pt; exact line recall is reported, not used as parser-parity evidence",
            "first_copy_headings": f"same-page normalized heading text recall >= {HEADING_RECALL_MIN:.2f}; tested against Python parser's accepted first-copy headings",
            "table_cells": f"glyph bbox center inside PyMuPDF cell rectangle, no geometry expansion; alpha-numeric cell character recall >= {CELL_CHAR_RECALL_MIN:.2f}",
            "frequency_associations": f"reaction-cell alphanumeric character multiset plus normalized frequency label (recognized English/Greek column headers; punctuation and order ignored) recall >= {FREQUENCY_ASSOCIATION_RECALL_MIN:.2f}; exact flattened-row text recall is separately reported",
            "bold_cells": f"PDFium font-weight >= {CELL_BOLD_THRESHOLD} is bold; compare against PyMuPDF page-span/table-cell overlap flags",
            "independent_oracle": "Xarelto EN expectations copied from tests/test_m1_acceptance.py:208-264; Anaemia/Common, Thrombocytosis/Uncommon, Eye haemorrhage/Common, Hypotension/Common; no Anaemia/Thrombocytosis glue; three SOC headings remain unbanded",
        },
        "summary": summary,
        "documents": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(args.output.name + ".part")
    temporary_output.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary_output.replace(args.output)
    print(json.dumps({"report": str(args.output), "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
