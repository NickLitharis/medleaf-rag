"""EMA source discovery: bulk medicine table, then per-language product-information PDFs.

SPEC section 5 says *no guessed URLs*. The flow is:

1. Read the medicine's EPAR page link from EMA's nightly bulk export
   (``medicines-output-medicines-report_en.xlsx``, sheet ``Medicine``).
2. Read the EPAR page and pull the EN and EL product-information PDF links from it.
3. Write the resolved URLs into ``data/medicines.yaml``.

Language trap (verified 2026-10-15): EMA's *English* document path serves every
language, e.g. ``/en/documents/product-information/<slug>-epar-product-information_el.pdf``
returns HTTP 301 to the **English** PDF. Content that looks Greek is therefore only
available under the matching locale prefix (``/el/documents/...``). The resolver keeps the
locale prefix from the page, and the downloader re-checks the detected language.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

import httpx

from medleaf.models import Lang

BULK_REPORT_URL = (
    "https://www.ema.europa.eu/en/documents/report/medicines-output-medicines-report_en.xlsx"
)
BULK_SHEET_NAME = "Medicine"
EPAR_LINK_COLUMN_INDEX = 39  # 1-based column in the 'Medicine' sheet; verified 2026-10-15.

# Matches the product-information PDF link for one language inside an EPAR page.
_PRODUCT_INFO_RE = re.compile(
    r'href="(?P<href>/(?P<locale>[a-z]{2})/documents/product-information/'
    r'(?P<slug>[a-z0-9-]+)-epar-product-information_(?P<lang>[a-z]{2})\.pdf)"',
    re.IGNORECASE,
)
_REVISION_META_RE = re.compile(
    r'<meta[^>]+(?:property|name)="(?:og:updated_time|article:modified_time)"[^>]+'
    r'content="(?P<date>\d{4}-\d{2}-\d{2})',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MedicineLink:
    """A medicine row located in EMA's bulk export."""

    name: str
    epar_url: str
    atc_code: str | None = None
    active_substance: str | None = None
    content_type: str | None = None


@dataclass(frozen=True)
class ResolvedDocument:
    """A resolved product-information PDF for one language."""

    lang: Lang
    url: str
    revision_date: str | None = None


def build_client(*, user_agent: str, timeout: float = 60.0) -> httpx.Client:
    """HTTP client that follows redirects and identifies the project to EMA."""
    return httpx.Client(
        headers={"User-Agent": user_agent},
        timeout=timeout,
        follow_redirects=True,
    )


def fetch_bulk_report(client: httpx.Client, url: str = BULK_REPORT_URL) -> bytes:
    """Download EMA's nightly bulk medicine export."""
    response = client.get(url)
    response.raise_for_status()
    return response.content


def parse_bulk_rows(content: bytes) -> list[MedicineLink]:
    """Parse the bulk XLSX into medicine rows.

    The report has a one-line preamble, so the header row is located by looking for the
    row whose first cell is ``Content type`` rather than assuming row 1.
    """
    rows = _read_xlsx_rows(content, sheet_name=BULK_SHEET_NAME)
    header_index = None
    for index, row in enumerate(rows):
        if row and row[0].strip().lower().rstrip(":").strip() == "content type":
            header_index = index
            break
    if header_index is None:
        raise ValueError("could not find the 'Content type' header row in the EMA bulk report")

    links: list[MedicineLink] = []
    for row in rows[header_index + 1 :]:
        if len(row) < 3:
            continue
        name = row[1].strip()
        epar_url = (
            row[EPAR_LINK_COLUMN_INDEX - 1].strip()
            if len(row) >= EPAR_LINK_COLUMN_INDEX
            else ""
        )
        if not name or not epar_url.lower().startswith("http"):
            continue
        links.append(
            MedicineLink(
                name=name,
                epar_url=epar_url,
                atc_code=_cell(row, 12),
                active_substance=_cell(row, 7),
                content_type=_cell(row, 1 - 1),
            )
        )
    return links


def _cell(row: list[str], index_1_based: int) -> str | None:
    if len(row) < index_1_based:
        return None
    value = row[index_1_based - 1].strip()
    return value or None


def _read_xlsx_rows(content: bytes, *, sheet_name: str) -> list[list[str]]:
    """Read one XLSX sheet as a dense grid of strings.

    Uses a minimal ZIP + XML reader so the ingestion dependency list stays small
    (PyMuPDF for PDFs, PyYAML for configs) instead of pulling in a spreadsheet library
    for one nightly report.
    """
    return _read_xlsx_rows_stdlib(content, sheet_name=sheet_name)


def _read_xlsx_rows_stdlib(content: bytes, *, sheet_name: str) -> list[list[str]]:
    import io
    import xml.etree.ElementTree as ET
    import zipfile

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{ns}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{ns}t")))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
        rel_targets = {
            rel.get("Id"): rel.get("Target")
            for rel in relationships.findall(f"{rel_ns}Relationship")
        }
        sheet_path = None
        for sheet in workbook.find(f"{ns}sheets") or []:
            if sheet.get("name") == sheet_name:
                rel_id = sheet.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                )
                target = rel_targets.get(rel_id or "", "")
                sheet_path = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
                break
        if sheet_path is None:
            raise ValueError(f"sheet {sheet_name!r} not found in the EMA bulk report")

        root = ET.fromstring(archive.read(sheet_path))
        grid: list[list[str]] = []
        for row in root.iter(f"{ns}row"):
            cells: dict[int, str] = {}
            for cell in row.findall(f"{ns}c"):
                ref = cell.get("r") or ""
                column = _column_index(ref)
                cell_type = cell.get("t")
                if cell_type == "inlineStr":
                    node = cell.find(f"{ns}is")
                    value = (
                        "".join(t.text or "" for t in node.iter(f"{ns}t"))
                        if node is not None
                        else ""
                    )
                else:
                    node = cell.find(f"{ns}v")
                    raw = node.text if node is not None and node.text else ""
                    if cell_type == "s":
                        value = shared[int(raw)] if raw.isdigit() and int(raw) < len(shared) else ""
                    else:
                        value = raw
                cells[column] = value
            if not cells:
                grid.append([])
                continue
            width = max(cells) + 1
            grid.append([cells.get(index, "") for index in range(width)])
        return grid


def _column_index(reference: str) -> int:
    """Convert an A1-style column reference to a zero-based index."""
    letters = "".join(char for char in reference if char.isalpha()).upper()
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return max(index - 1, 0)


def parse_product_information_links(html: str) -> dict[Lang, str]:
    """Pull EN and EL product-information PDF URLs from an EPAR page.

    Prefers the link whose locale prefix matches the document language, because the
    ``/en/`` path silently redirects non-English documents to English.
    """
    found: dict[Lang, str] = {}
    for match in _PRODUCT_INFO_RE.finditer(html):
        locale = match.group("locale").lower()
        lang = match.group("lang").lower()
        if lang not in ("en", "el"):
            continue
        url = f"https://www.ema.europa.eu{match.group('href')}"
        preferred = locale == lang
        if lang not in found or preferred:
            if lang in found and not preferred:
                continue
            found[Lang(lang)] = url
    return found


def parse_revision_date(html: str) -> str | None:
    """Read the EPAR page's last-updated date, used as a fallback revision marker."""
    match = _REVISION_META_RE.search(html)
    return match.group("date") if match else None


def resolve_medicine(
    client: httpx.Client,
    epar_url: str,
) -> tuple[dict[Lang, ResolvedDocument], str | None]:
    """Fetch an EPAR page and resolve its EN and EL product-information PDFs."""
    response = client.get(epar_url)
    response.raise_for_status()
    html = response.text
    links = parse_product_information_links(html)
    revision = parse_revision_date(html)
    documents = {
        lang: ResolvedDocument(lang=lang, url=url, revision_date=revision)
        for lang, url in links.items()
    }
    return documents, revision


def main(argv: list[str] | None = None) -> int:
    """Print the bulk-report row for each requested medicine name."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m medleaf.ingest.ema <Medicine name> [...]", file=sys.stderr)
        return 2
    from medleaf.config import get_settings

    settings = get_settings()
    with build_client(user_agent=settings.user_agent, timeout=settings.request_timeout_s) as client:
        rows = parse_bulk_rows(fetch_bulk_report(client))
    by_name = {row.name.lower(): row for row in rows}
    for name in argv:
        row = by_name.get(name.lower())
        print(f"{name}: {row.epar_url if row else 'NOT FOUND'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
