"""``medleaf`` command line: resolve, download, parse, index and search.

Commands are deliberately small and composable; each prints a summary that is safe to
paste into a PR, and exits non-zero when a real problem is found. Retrieval commands
import their heavy dependencies (Qdrant client, torch) only inside the command, so
`medleaf check` keeps working without them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:  # imported lazily so non-retrieval commands work without them
    from qdrant_client import QdrantClient

    from medleaf.retrieval.embed import Encoder

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


def _qdrant_client(settings: Settings) -> QdrantClient:
    """Create a Qdrant client, failing with a clear message rather than an import error."""
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=int(settings.request_timeout_s),
    )


def _encoder_or_none(settings: Settings) -> Encoder | None:
    """Load the pinned embedding model, printing a clear error when it is unavailable."""
    from medleaf.retrieval.embed import E5Encoder, EncoderError

    try:
        encoder = E5Encoder.load(
            model_id=settings.embed_model,
            revision=settings.embed_revision,
            device=settings.embed_device,
            batch_size=settings.embed_batch_size,
            max_seq_length=settings.embed_max_seq_length,
            cache_dir=settings.embed_cache_dir,
        )
    except EncoderError as error:
        print(f"embedding model unavailable: {error}", file=sys.stderr)
        return None
    print(
        f"model {encoder.model_id}@{encoder.revision} loaded (dim {encoder.dimension})",
        file=sys.stderr,
    )
    return encoder


def _cmd_index_build(args: argparse.Namespace, settings: Settings) -> int:
    """Parse the corpus, embed every chunk and upsert it into Qdrant."""
    from medleaf.retrieval.pipeline import IndexBuildError, build_index

    encoder = _encoder_or_none(settings)
    if encoder is None:
        return 2

    client = _qdrant_client(settings)
    try:
        client.get_collections()
    except Exception as error:
        print(f"Qdrant unavailable at {settings.qdrant_url}: {error}", file=sys.stderr)
        return 2

    try:
        manifest = build_index(
            settings=settings,
            client=client,
            encoder=encoder,
            only=args.only or None,
            recreate=not args.no_recreate,
            reuse_cache=not args.no_cache,
            allow_missing=args.allow_missing,
            allow_checksum_mismatch=args.allow_checksum_mismatch,
        )
    except IndexBuildError as error:
        print(f"index build failed: {error}", file=sys.stderr)
        return 1
    print(
        f"built {manifest.total_chunks} chunks across {len(manifest.chunk_counts)} "
        f"medicine-language pairs; corpus_version={manifest.corpus_version}; "
        f"missing={manifest.missing}"
    )
    return 0


def _cmd_index_window_report(args: argparse.Namespace, settings: Settings) -> int:
    """Measure every indexed chunk against the model window (no re-embedding)."""
    from medleaf.retrieval.pipeline import window_report

    encoder = _encoder_or_none(settings)
    if encoder is None:
        return 2
    client = _qdrant_client(settings)
    try:
        client.get_collections()
    except Exception as error:
        print(f"Qdrant unavailable at {settings.qdrant_url}: {error}", file=sys.stderr)
        return 2
    report = window_report(
        settings=settings, client=client, encoder=encoder, worst_limit=args.worst
    )
    if args.json:
        print(
            json.dumps(
                {
                    "collection": report.collection,
                    "points": report.total_points,
                    "over_window": report.over_window,
                    "clean": report.clean,
                    "limit_tokens": report.limit_tokens,
                    "by_medicine": report.by_medicine,
                    "worst": [
                        {"chunk_id": chunk_id, "tokens": count}
                        for chunk_id, count in report.worst
                    ],
                },
                indent=2,
            )
        )
        return 0 if report.clean else 1
    print(
        f"collection={report.collection} points={report.total_points} "
        f"over_window={report.over_window} limit_tokens={report.limit_tokens}"
    )
    for chunk_id, count in report.worst:
        print(f"  worst: {count:5} tokens  {chunk_id}")
    if report.by_medicine:
        print(f"  medicines with over-window chunks: {report.by_medicine}")
    return 0 if report.clean else 1


def _cmd_index_status(args: argparse.Namespace, settings: Settings) -> int:
    """Report the index manifest and the live readiness probe."""
    from medleaf.retrieval.index import IndexMissingError
    from medleaf.retrieval.service import RetrievalService

    service = RetrievalService(settings)
    try:
        manifest = service.manifest()
    except IndexMissingError as error:
        print(f"index missing: {error}", file=sys.stderr)
        return 1
    print(manifest.model_dump_json(indent=2))
    state = service.ready()
    print(f"ready={state.ready}: {state.detail}")
    return 0 if state.ready else 1


def _cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    """Run hybrid retrieval from the command line (same path as ``/v1/search``)."""
    from medleaf.retrieval.index import IndexMissingError, RetrievalUnavailableError
    from medleaf.retrieval.service import RetrievalService

    service = RetrievalService(settings)
    try:
        response = service.search(
            args.query,
            lang=args.lang,
            medicine=args.medicine or None,
            doc=args.doc,
            active_substance=args.active_substance,
            corpus_version=args.corpus_version,
            limit=args.limit,
        )
    except (IndexMissingError, RetrievalUnavailableError) as error:
        print(f"search unavailable: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(response.model_dump_json(indent=2))
        return 0
    print(
        f"query={response.query!r} lang={response.lang} corpus_version={response.corpus_version} "
        f"count={response.count} latency_ms={response.latency_ms}"
    )
    for hit in response.results:
        snippet = " ".join(hit.text.split())[:160]
        substance = hit.active_substance or "substance: unavailable"
        print(
            f"{hit.rank:2}. {hit.score:.4f} {hit.chunk_id} [{hit.lang}/{hit.doc}] "
            f"{substance} ({hit.atc_code or 'atc: unavailable'}) {snippet}"
        )
        print(f"    source: {hit.source_url} (revision {hit.revision_date})")
    return 0


def _cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    """Score one split, write the result file and optionally gate it on a baseline."""
    from medleaf.eval.runner import TOP_K, evaluate, gate, load_questions, write_result
    from medleaf.retrieval.index import IndexMissingError, RetrievalUnavailableError
    from medleaf.retrieval.service import RetrievalService

    service = RetrievalService(settings)

    def search(query: str) -> list[str]:
        return [hit.chunk_id for hit in service.search(query, limit=TOP_K).results]

    questions = load_questions(args.dataset, args.split)
    try:
        result = evaluate(questions, search)
    except (IndexMissingError, RetrievalUnavailableError) as error:
        print(f"eval aborted, retrieval unavailable: {error}", file=sys.stderr)
        return 2
    manifest = service.manifest()
    result = {
        "split": args.split,
        "dataset": args.dataset.as_posix(),
        "corpus_version": manifest.corpus_version,
        "embed_model": f"{manifest.embed_model}@{manifest.embed_revision}",
        **result,
    }
    path = write_result(result, args.out, args.split)
    overall = result["overall"]
    print(
        f"{args.split}: Recall@5 {overall['recall_at_5']['hits']}/{overall['total']} "
        f"({overall['recall_at_5']['rate']:.2%}), Recall@10 "
        f"{overall['recall_at_10']['hits']}/{overall['total']}, MRR@10 {overall['mrr_at_10']}"
    )
    for lang, row in result["by_lang"].items():
        print(f"  {lang}: Recall@5 {row['recall_at_5']['hits']}/{row['total']}")
    print(f"result: {path}")
    if args.baseline is None:
        return 0
    failures = gate(result, json.loads(args.baseline.read_text(encoding="utf-8")))
    for failure in failures:
        print(f"GATE FAILED: {failure}", file=sys.stderr)
    if not failures:
        print(f"gate passed against {args.baseline}")
    return 1 if failures else 0


def _cmd_index_annotate(args: argparse.Namespace, settings: Settings) -> int:
    """Attach SPEC section 8 payload metadata to an existing index.

    Payload-only: the vectors are left alone, so enriching metadata never triggers a
    re-embed of an otherwise correct index.
    """
    from medleaf.retrieval.pipeline import IndexBuildError, annotate_manifest_metadata

    client = _qdrant_client(settings)
    try:
        client.get_collections()
    except Exception as error:
        print(f"Qdrant unavailable at {settings.qdrant_url}: {error}", file=sys.stderr)
        return 2
    try:
        updated = annotate_manifest_metadata(settings=settings, client=client)
    except IndexBuildError as error:
        print(f"metadata annotate failed: {error}", file=sys.stderr)
        return 1
    print(
        f"updated metadata for {len(updated)} medicines "
        f"({sum(updated.values())} points updated in payload only)"
    )
    return 0


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

    index = subparsers.add_parser("index", help="build and inspect the Qdrant index")
    index_sub = index.add_subparsers(dest="index_command", required=True)
    index_build = index_sub.add_parser("build", help="parse, embed and upsert the corpus")
    index_build.add_argument("--only", nargs="*", default=[], help="limit to these medicine slugs")
    index_build.add_argument("--no-cache", action="store_true", help="re-parse every PDF")
    index_build.add_argument(
        "--no-recreate", action="store_true", help="upsert into the existing collection"
    )
    index_build.add_argument(
        "--allow-missing",
        action="store_true",
        help="record missing PDFs in the manifest instead of aborting",
    )
    index_build.add_argument(
        "--allow-checksum-mismatch",
        action="store_true",
        help="index PDFs whose SHA-256 does not match the pin in data/medicines.yaml",
    )
    index_build.set_defaults(func=_cmd_index_build)
    index_status = index_sub.add_parser("status", help="print the index manifest and readiness")
    index_status.set_defaults(func=_cmd_index_status)
    index_sub.add_parser(
        "annotate",
        help="attach chunk metadata to an existing index (payload only, no re-embed)",
    ).set_defaults(func=_cmd_index_annotate)
    index_window = index_sub.add_parser(
        "window-report",
        help="measure indexed chunks against the model window (no re-embed)",
    )
    index_window.add_argument("--worst", type=int, default=10, help="worst offenders to list")
    index_window.add_argument("--json", action="store_true", help="print machine-readable output")
    index_window.set_defaults(func=_cmd_index_window_report)

    search = subparsers.add_parser("search", help="hybrid search over the built index")
    search.add_argument("query", help="question or keyword phrase")
    search.add_argument("--lang", choices=["en", "el"], default=None, help="corpus language")
    search.add_argument("--medicine", nargs="*", default=[], help="medicine slug filter")
    search.add_argument("--doc", choices=["smpc", "leaflet"], default=None, help="document filter")
    search.add_argument(
        "--active-substance", default=None, help="filter chunks by active substance"
    )
    search.add_argument("--corpus-version", default=None, help="exact corpus version filter")
    search.add_argument("--limit", type=int, default=10, help="number of chunks to return")
    search.add_argument("--json", action="store_true", help="print the full JSON response")
    search.set_defaults(func=_cmd_search)

    evaluate = subparsers.add_parser("eval", help="retrieval eval on a split (SPEC §11)")
    evaluate.add_argument("--split", choices=["dev", "test"], required=True)
    evaluate.add_argument(
        "--dataset", type=Path, default=Path("eval/datasets/retrieval-v1.jsonl")
    )
    evaluate.add_argument("--out", type=Path, default=Path("eval/results"))
    evaluate.add_argument(
        "--baseline", type=Path, help="fail (exit 1) if Recall@5 regresses against this file"
    )
    evaluate.set_defaults(func=_cmd_eval)
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
