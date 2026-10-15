"""``medleaf`` command line: resolve corpus URLs, download PDFs, report coverage.

Commands are deliberately small and composable; each prints a summary that is safe to
paste into a PR, and exits non-zero when a real problem is found.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from medleaf.config import Settings, get_settings
from medleaf.ingest import ema as ema_module
from medleaf.ingest.chunk import is_readable_frequency_row
from medleaf.ingest.download import sha256_of
from medleaf.ingest.ema import build_client
from medleaf.ingest.pipeline import parse_pdf
from medleaf.ingest.store import dump_medicines, load_medicines, slugify, upsert_medicine
from medleaf.models import DocKind, Lang, Medicine
from medleaf.sections import LEAFLET_REQUIRED, SMPC_REQUIRED


@dataclass
class CoverageRow:
    """Per medicine/language coverage of the sections M1 requires."""

    medicine: str
    lang: str
    smpc_found: int
    leaflet_found: int
    missing_smpc: list[str]
    missing_leaflet: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_smpc and not self.missing_leaflet


def _cmd_resolve(args: argparse.Namespace, settings: Settings) -> int:
    """Resolve EN/EL PDF URLs for the requested medicine names and update medicines.yaml."""
    with build_client(user_agent=settings.user_agent, timeout=settings.request_timeout_s) as client:
        rows = ema_module.parse_bulk_rows(ema_module.fetch_bulk_report(client))
        by_name = {row.name.lower(): row for row in rows}
        medicines = load_medicines(settings.medicines_file)

        for name in args.names:
            row = by_name.get(name.lower())
            if row is None:
                print(f"NOT FOUND in EMA bulk report: {name}", file=sys.stderr)
                return 1
            documents, revision = ema_module.resolve_medicine(client, row.epar_url)
            missing = {Lang.EL, Lang.EN} - set(documents)
            if missing:
                names = ", ".join(sorted(lang.value for lang in missing))
                print(f"{name}: no product-information PDF for {names}", file=sys.stderr)
                return 1
            medicine = Medicine(
                slug=slugify(row.name),
                name=row.name,
                active_substance=row.active_substance or "",
                atc_code=row.atc_code,
                epar_url=row.epar_url,
                documents={
                    lang: {
                        "lang": lang.value,
                        "url": document.url,
                        "revision_date": document.revision_date or revision,
                    }
                    for lang, document in documents.items()
                },
            )
            medicines = upsert_medicine(medicines, medicine)
            print(f"resolved {name}:")
            for lang, document in sorted(documents.items(), key=lambda item: item[0].value):
                print(f"  {lang.value}: {document.url}")

    dump_medicines(settings.medicines_file, medicines)
    print(f"wrote {settings.medicines_file} ({len(medicines)} medicines)")
    return 0


def _cmd_download(args: argparse.Namespace, settings: Settings) -> int:
    """Download every PDF in medicines.yaml, then pin checksums back into the file."""
    medicines = load_medicines(settings.medicines_file)
    if not medicines:
        print(
            f"no medicines in {settings.medicines_file}; run `medleaf resolve` first",
            file=sys.stderr,
        )
        return 1

    changed = 0
    mismatched: list[str] = []
    with build_client(user_agent=settings.user_agent, timeout=settings.request_timeout_s) as client:
        for medicine in medicines:
            for lang, document in sorted(medicine.documents.items(), key=lambda i: i[0].value):
                destination = settings.pdf_path(medicine.slug, lang.value)
                try:
                    from medleaf.ingest.download import download_pdf

                    result = download_pdf(
                        client,
                        url=str(document.url),
                        destination=destination,
                        expected_sha256=document.sha256,
                        lang=lang,
                    )
                except Exception as error:  # reported, not hidden
                    print(f"FAIL {medicine.slug} {lang.value}: {error}", file=sys.stderr)
                    return 1
                print(
                    f"{medicine.slug:10} {lang.value}: {result.status} "
                    f"({result.bytes_written} bytes, sha256 {result.sha256[:12]}...)"
                )
                if result.changed:
                    changed += 1
                if result.detected_lang is not None and result.detected_lang is not lang:
                    mismatched.append(f"{medicine.slug}/{lang.value}")
                document.sha256 = result.sha256

    if not args.no_pin:
        dump_medicines(settings.medicines_file, medicines)
        print(f"pinned checksums in {settings.medicines_file}")
    if changed:
        print(f"{changed} document(s) changed since medicines.yaml was pinned (new corpus version)")
    if mismatched:
        print("language mismatch for: " + ", ".join(mismatched), file=sys.stderr)
        return 1
    return 0


def _cmd_check(args: argparse.Namespace, settings: Settings) -> int:
    """Parse the downloaded corpus and report section coverage per medicine and language."""
    medicines = load_medicines(settings.medicines_file)
    rows: list[CoverageRow] = []
    for medicine in medicines:
        if args.only and medicine.slug not in args.only:
            continue
        for lang in (Lang.EN, Lang.EL):
            pdf_path = settings.pdf_path(medicine.slug, lang.value)
            if not pdf_path.exists():
                print(f"missing {pdf_path}; run `medleaf download` first", file=sys.stderr)
                return 1
            parsed = parse_pdf(
                medicine=medicine,
                lang=lang,
                pdf_path=pdf_path,
                corpus_version=settings.corpus_version,
                max_tokens=settings.max_section_tokens,
            )
            smpc = list(
                dict.fromkeys(section.section for section in parsed.by_doc(DocKind.SMPC))
            )
            leaflet = list(
                dict.fromkeys(section.section for section in parsed.by_doc(DocKind.LEAFLET))
            )
            rows.append(
                CoverageRow(
                    medicine=medicine.slug,
                    lang=lang.value,
                    smpc_found=len(smpc),
                    leaflet_found=len(leaflet),
                    missing_smpc=[key for key in SMPC_REQUIRED if key not in smpc],
                    missing_leaflet=[key for key in LEAFLET_REQUIRED if key not in leaflet],
                )
            )

    width = max((len(row.medicine) for row in rows), default=8)
    for row in rows:
        flag = "ok  " if row.ok else "FAIL"
        detail = ""
        if not row.ok:
            detail = f" missing smpc={row.missing_smpc} leaflet={row.missing_leaflet}"
        print(
            f"{flag} {row.medicine:<{width}} {row.lang} "
            f"smpc={row.smpc_found:2} leaflet={row.leaflet_found}{detail}"
        )

    failures = [row for row in rows if not row.ok]
    print(f"{len(rows) - len(failures)}/{len(rows)} medicine-language pairs complete")
    return 1 if failures else 0


def _cmd_tables(args: argparse.Namespace, settings: Settings) -> int:
    """Print representative flattened 4.8 side-effect table rows per medicine/language.

    This is the M1 readability check: it proves the table flattener produced rows with
    an adverse reaction and a frequency, not merely that tables were detected.
    """
    medicines = load_medicines(settings.medicines_file)
    printed = 0
    have_rows = 0
    for medicine in medicines:
        if args.only and medicine.slug not in args.only:
            continue
        for lang in (Lang.EN, Lang.EL):
            pdf_path = settings.pdf_path(medicine.slug, lang.value)
            if not pdf_path.exists():
                print(f"missing {pdf_path}; run `medleaf download` first", file=sys.stderr)
                return 1
            parsed = parse_pdf(
                medicine=medicine,
                lang=lang,
                pdf_path=pdf_path,
                corpus_version=settings.corpus_version,
                max_tokens=settings.max_section_tokens,
            )
            lines = [
                line
                for section in parsed.by_doc(DocKind.SMPC)
                if section.section == "4.8"
                for line in section.text.splitlines()
                if is_readable_frequency_row(line)
            ]
            printed += 1
            if lines:
                have_rows += 1
            print(f"{medicine.slug} {lang.value}: 4.8 frequency rows={len(lines)}")
            for sample in lines[: args.samples]:
                shown = (
                    sample if len(sample) <= args.max_chars else sample[: args.max_chars] + "..."
                )
                print(f"    {shown}")
    print(f"{have_rows}/{printed} medicine-language pairs have readable 4.8 frequency rows")
    return 0 if have_rows == printed and printed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medleaf", description="MedLeaf RAG ingestion CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve", help="resolve EN/EL PDF URLs from EMA")
    resolve.add_argument("names", nargs="+", help="EMA medicine names, e.g. Eliquis")
    resolve.set_defaults(func=_cmd_resolve)

    download = subparsers.add_parser("download", help="download PDFs listed in medicines.yaml")
    download.add_argument(
        "--no-pin",
        action="store_true",
        help="do not write downloaded checksums back into medicines.yaml",
    )
    download.set_defaults(func=_cmd_download)

    check = subparsers.add_parser("check", help="report section coverage for the local corpus")
    check.add_argument("--only", nargs="*", default=[], help="limit to these medicine slugs")
    check.set_defaults(func=_cmd_check)

    tables = subparsers.add_parser("tables", help="print readable 4.8 side-effect table rows")
    tables.add_argument("--only", nargs="*", default=[], help="limit to these medicine slugs")
    tables.add_argument("--samples", type=int, default=4, help="rows to show per medicine/language")
    tables.add_argument(
        "--max-chars", type=int, default=200, help="truncate each sample row to this length"
    )
    tables.set_defaults(func=_cmd_tables)
    return parser


def main(argv: list[str] | None = None) -> int:
    # EMA text is Greek/UTF-8; a Windows console defaults to cp1252 and would crash on
    # frequency symbols like '≥'. Force UTF-8 output so reports are pasteable.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
        pass
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    result: int = args.func(args, settings)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# Re-exported for tests that assert checksum behaviour without a network round trip.
__all__ = ["CoverageRow", "main", "sha256_of"]
