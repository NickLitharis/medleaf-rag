"""Download product-information PDFs into ``data/raw`` and verify they are the right language.

The checksum rule from SPEC section 5: the repository stores a SHA-256 per file, so a
mismatch means EMA published a new corpus version. That is reported, not raised.

EMA rate-limits bursts (HTTP 429), so :func:`download_pdf` retries with exponential
backoff and honours ``Retry-After``. The language re-check matters for the same reason
it did in M1: a document can be served from the wrong locale path, and only the bytes
prove which language arrived.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from medleaf.ingest.pdf import detect_language, extract_lines
from medleaf.models import Lang, Medicine

#: Status codes worth retrying; EMA returns 429 under burst load.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 30.0


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of one language's download."""

    slug: str
    lang: Lang
    path: Path
    sha256: str
    status: str
    expected_sha256: str | None = None
    bytes_written: int = 0
    detected_lang: Lang | None = None

    @property
    def changed(self) -> bool:
        """True when the download does not match the checksum recorded in medicines.yaml."""
        if self.expected_sha256 is None:
            return False
        return self.expected_sha256 != self.sha256


def sha256_of(path: Path) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_pdf(
    client: httpx.Client,
    *,
    url: str,
    destination: Path,
    expected_sha256: str | None = None,
    lang: Lang | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> DownloadResult:
    """Download one PDF, verify its language when asked, and hash it.

    Retries throttled or transiently-failing responses; a persistent 4xx other than 429
    is raised immediately, because retrying a wrong URL only wastes EMA's bandwidth.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = _get_with_retries(
        client, url, max_attempts=max_attempts, backoff_seconds=backoff_seconds
    )
    content_type = response.headers.get("content-type", "")
    if "pdf" not in content_type.lower():
        raise ValueError(
            f"{url} returned content-type {content_type!r}, not a PDF; "
            "EMAIL likely moved the document"
        )

    temp_path = destination.with_suffix(".pdf.part")
    temp_path.write_bytes(response.content)
    temp_path.replace(destination)

    digest = sha256_of(destination)
    detected: Lang | None = None
    status = "downloaded"
    if lang is not None:
        detected = _detect_pdf_language(destination)
        if detected is not lang:
            status = f"language mismatch: requested {lang.value}, detected {detected.value}"
        else:
            status = "verified"

    result = DownloadResult(
        slug=destination.parent.name,
        lang=lang or Lang.EN,
        path=destination,
        sha256=digest,
        status=status,
        expected_sha256=expected_sha256,
        bytes_written=len(response.content),
        detected_lang=detected,
    )
    if result.changed:
        return DownloadResult(
            slug=result.slug,
            lang=result.lang,
            path=result.path,
            sha256=result.sha256,
            status=f"checksum changed (corpus version differs from medicines.yaml): {status}",
            expected_sha256=expected_sha256,
            bytes_written=result.bytes_written,
            detected_lang=detected,
        )
    return result


def _get_with_retries(
    client: httpx.Client,
    url: str,
    *,
    max_attempts: int,
    backoff_seconds: float,
) -> httpx.Response:
    """GET a URL, retrying throttled/transient responses with exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        response: httpx.Response | None = None
        try:
            response = client.get(url)
        except httpx.TransportError:  # timeouts, connection resets
            if attempt == max_attempts:
                raise
        else:
            if response.status_code not in RETRY_STATUS:
                response.raise_for_status()
                return response
            if attempt == max_attempts:
                response.raise_for_status()
        delay = min(backoff_seconds * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
        retry_after = _retry_after_seconds(None if response is None else response.headers)
        time.sleep(retry_after if retry_after is not None else delay)
    raise AssertionError("retry loop exhausted without returning or raising")  # pragma: no cover


def _retry_after_seconds(headers: httpx.Headers | None) -> float | None:
    """Read a numeric ``Retry-After`` when EMA sends one."""
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return min(float(raw), MAX_BACKOFF_SECONDS)
    except ValueError:
        return None


def _detect_pdf_language(path: Path) -> Lang:
    """Detect a PDF's language from its first pages, using the same rule as the parser."""
    lines = extract_lines(str(path), max_pages=6)
    sample = "\n".join(line.text for line in lines)
    return detect_language(sample)


def download_medicine(
    client: httpx.Client, medicine: Medicine, raw_dir: Path
) -> list[DownloadResult]:
    """Download both language versions of one medicine."""
    results: list[DownloadResult] = []
    for lang, document in medicine.documents.items():
        destination = raw_dir / medicine.slug / f"{medicine.slug}_{lang.value}.pdf"
        results.append(
            download_pdf(
                client,
                url=str(document.url),
                destination=destination,
                expected_sha256=document.sha256,
                lang=lang,
            )
        )
    return results


__all__ = [
    "DEFAULT_BACKOFF_SECONDS",
    "DEFAULT_MAX_ATTEMPTS",
    "MAX_BACKOFF_SECONDS",
    "RETRY_STATUS",
    "DownloadResult",
    "download_medicine",
    "download_pdf",
    "sha256_of",
]
