"""``medleaf`` command line: resolve, download, check and inspect the EMA corpus.

Commands are deliberately small and composable; each prints a summary that is safe to
paste into a PR, and exits non-zero when a real problem is found. Embedding and the
clustering comparison live in ``experiments/e-means``.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from pydantic import HttpUrl

from medleaf.config import Settings, get_settings
from medleaf.ingest import ema as ema_module
from medleaf.ingest.chunk import is_readable_frequency_row
from medleaf.ingest.download import DownloadResult, download_pdf, sha256_of
from medleaf.ingest.ema import build_client
from medleaf.ingest.pipeline import parse_pdf
from medleaf.ingest.store import dump_medicines, load_medicines, slugify, upsert_medicine
from medleaf.models import DocKind, DocumentRef, Lang, Medicine
from medleaf.sections import LEAFLET_REQUIRED, SMPC_REQUIRED

#: Concurrent EMA downloads. Small on purpose: EMA rate-limits bursts with HTTP 429.
DEFAULT_DOWNLOAD_WORKERS = 2


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


def _ema_cache_dir(settings: Settings, override: str | None) -> Path:
    return Path(override) if override else settings.data_dir / "cache" / "ema"


def _kept_pin(
    medicines: list[Medicine], slug: str, lang: Lang, url: str
) -> str | None:
    """Return the existing checksum when the resolved URL is unchanged."""
    for medicine in medicines:
        if medicine.slug != slug or lang not in medicine.documents:
            continue
        existing = medicine.documents[lang]
        if str(existing.url) == url:
            return existing.sha256
    return None


def _cmd_resolve(args: argparse.Namespace, settings: Settings) -> int:
    """Resolve EN/EL PDF URLs for the requested medicine names and update medicines.yaml.

    ``--continue-on-error`` is what makes a ~40-medicine batch practical: a name that
    EMA has withdrawn is reported and skipped instead of discarding the rest of the run.
    """
    cache_dir = _ema_cache_dir(settings, args.cache_dir)
    medicines = load_medicines(settings.medicines_file)
    if not args.names and not args.drop:
        print("nothing to do: pass medicine names and/or --drop", file=sys.stderr)
        return 2
    if args.drop:
        wanted = set(args.drop)
        dropped = [medicine.slug for medicine in medicines if medicine.slug in wanted]
        medicines = [medicine for medicine in medicines if medicine.slug not in wanted]
        for slug in dropped:
            print(f"dropped {slug} from {settings.medicines_file}")
        absent = sorted(wanted - set(dropped))
        if absent:
            print(f"not present, nothing to drop: {', '.join(absent)}")
    if not args.names:
        dump_medicines(settings.medicines_file, medicines)
        print(f"wrote {settings.medicines_file} ({len(medicines)} medicines)")
        return 0

    failures: list[str] = []
    resolved = 0

    with build_client(user_agent=settings.user_agent, timeout=settings.request_timeout_s) as client:
        rows = ema_module.parse_bulk_rows(
            ema_module.fetch_bulk_report_cached(client, cache_dir, force=args.refresh)
        )
        by_name = {row.name.lower(): row for row in rows}

        for position, name in enumerate(args.names):
            row = by_name.get(name.lower())
            if row is None:
                print(f"NOT FOUND in EMA bulk report: {name}", file=sys.stderr)
                failures.append(name)
                if not args.continue_on_error:
                    break
                continue
            try:
                documents, revision = ema_module.resolve_medicine_cached(
                    client, row.epar_url, cache_dir=cache_dir, force=args.refresh
                )
            except Exception as error:  # network or HTTP failure: report, do not hide
                print(f"FAILED to read EPAR page for {name}: {error}", file=sys.stderr)
                failures.append(name)
                if not args.continue_on_error:
                    break
                continue
            missing = {Lang.EL, Lang.EN} - set(documents)
            if missing:
                names = ", ".join(sorted(lang.value for lang in missing))
                print(f"{name}: no product-information PDF for {names}", file=sys.stderr)
                failures.append(name)
                if not args.continue_on_error:
                    break
                continue
            medicines = upsert_medicine(
                medicines,
                Medicine(
                    slug=slugify(row.name),
                    name=row.name,
                    active_substance=row.active_substance or "",
                    atc_code=row.atc_code,
                    epar_url=HttpUrl(row.epar_url),
                    documents={
                        lang: DocumentRef(
                            lang=lang,
                            url=HttpUrl(document.url),
                            revision_date=document.revision_date or revision,
                            # Re-resolving an unchanged URL keeps the pinned checksum, so a
                            # routine re-resolve cannot silently unpin the corpus. A moved
                            # document loses its pin and is re-downloaded and re-pinned.
                            sha256=_kept_pin(medicines, slugify(row.name), lang, document.url),
                        )
                        for lang, document in documents.items()
                    },
                ),
            )
            resolved += 1
            print(f"resolved {name}:")
            for lang, document in sorted(documents.items(), key=lambda item: item[0].value):
                print(f"  {lang.value}: {document.url}")
            if args.delay and position + 1 < len(args.names):
                time.sleep(args.delay)

    dump_medicines(settings.medicines_file, medicines)
    print(
        f"wrote {settings.medicines_file} ({len(medicines)} medicines); "
        f"resolved {resolved}, failed {len(failures)}"
    )
    if failures:
        print("unresolved: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


def _download_job(
    client: object,
    job: tuple[Medicine, Lang, DocumentRef, Path],
    *,
    force: bool,
) -> DownloadResult:
    """Download one PDF unless the pinned checksum already matches the local file."""
    medicine, lang, document, destination = job
    if not force and destination.exists() and document.sha256:
        digest = sha256_of(destination)
        if digest == document.sha256:
            return DownloadResult(
                slug=medicine.slug,
                lang=lang,
                path=destination,
                sha256=digest,
                status="unchanged (matches pinned checksum)",
                expected_sha256=document.sha256,
                bytes_written=destination.stat().st_size,
                detected_lang=lang,
            )
    return download_pdf(
        client,  # type: ignore[arg-type]
        url=str(document.url),
        destination=destination,
        expected_sha256=document.sha256,
        lang=lang,
    )


def _cmd_download(args: argparse.Namespace, settings: Settings) -> int:
    """Download every PDF in medicines.yaml, then pin checksums back into the file."""
    medicines = load_medicines(settings.medicines_file)
    if not medicines:
        print(
            f"no medicines in {settings.medicines_file}; run `medleaf resolve` first",
            file=sys.stderr,
        )
        return 1

    jobs: list[tuple[Medicine, Lang, DocumentRef, Path]] = [
        (medicine, lang, document, settings.pdf_path(medicine.slug, lang.value))
        for medicine in medicines
        for lang, document in sorted(medicine.documents.items(), key=lambda item: item[0].value)
    ]

    changed = 0
    mismatched: list[str] = []
    failures: list[str] = []
    results: list[DownloadResult] = []
    with (
        build_client(user_agent=settings.user_agent, timeout=settings.request_timeout_s) as client,
        ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool,
    ):
        futures = [
            (job, pool.submit(_download_job, client, job, force=args.force)) for job in jobs
        ]
        for job, future in futures:
            medicine, lang, _, _ = job
            try:
                result = future.result()
            except Exception as error:  # reported, not hidden
                print(f"FAIL {medicine.slug} {lang.value}: {error}", file=sys.stderr)
                failures.append(f"{medicine.slug}/{lang.value}")
                continue
            results.append(result)
            if result.changed:
                changed += 1

    results.sort(key=lambda item: (item.slug, item.lang.value))
    for result in results:
        print(
            f"{result.slug:14} {result.lang.value}: {result.status} "
            f"({result.bytes_written} bytes, sha256 {result.sha256[:12]}...)"
        )
        if result.detected_lang is not None and result.detected_lang is not result.lang:
            mismatched.append(f"{result.slug}/{result.lang.value}")

    if not args.no_pin:
        by_slug = {medicine.slug: medicine for medicine in medicines}
        for result in results:
            medicine = by_slug[result.slug]
            medicine.documents[result.lang] = medicine.documents[result.lang].model_copy(
                update={"sha256": result.sha256}
            )
        dump_medicines(settings.medicines_file, medicines)
        print(f"pinned checksums in {settings.medicines_file}")
    if changed:
        print(f"{changed} document(s) changed since medicines.yaml was pinned (new corpus version)")
    if mismatched:
        print("language mismatch for: " + ", ".join(mismatched), file=sys.stderr)
        return 1
    if failures:
        print(f"{len(failures)} download(s) failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


def _cmd_check(args: argparse.Namespace, settings: Settings) -> int:
    """Parse the downloaded corpus and report section coverage per medicine and language."""
    medicines = load_medicines(settings.medicines_file)
    if not medicines:
        print(
            f"no medicines in {settings.medicines_file}; "
            "run `medleaf resolve` or `medleaf download` first",
            file=sys.stderr,
        )
        return 1
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
    if not rows:
        print(
            "no medicine-language rows to check; check the --only slugs against "
            f"{settings.medicines_file}",
            file=sys.stderr,
        )
        return 1
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
    resolve.add_argument(
        "names",
        nargs="*",
        default=[],
        help="EMA medicine names, e.g. Eliquis (omit when only --drop is used)",
    )
    resolve.add_argument(
        "--drop",
        nargs="*",
        default=[],
        help="remove these medicine slugs from medicines.yaml (reported, never silent)",
    )
    resolve.add_argument(
        "--continue-on-error",
        action="store_true",
        help="keep resolving after a name that EMA has withdrawn; still exits 1",
    )
    resolve.add_argument("--delay", type=float, default=0.5, help="seconds between EPAR pages")
    resolve.add_argument("--cache-dir", default=None, help="override the EMA response cache")
    resolve.add_argument("--refresh", action="store_true", help="ignore the cached EMA responses")
    resolve.set_defaults(func=_cmd_resolve)

    download = subparsers.add_parser("download", help="download PDFs listed in medicines.yaml")
    download.add_argument(
        "--no-pin",
        action="store_true",
        help="do not write downloaded checksums back into medicines.yaml",
    )
    download.add_argument(
        "--workers", type=int, default=DEFAULT_DOWNLOAD_WORKERS, help="concurrent downloads"
    )
    download.add_argument("--force", action="store_true", help="re-download matching files")
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
