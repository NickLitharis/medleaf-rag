"""Download product-information PDFs into ``data/raw`` and verify they are the right language.

The checksum rule from SPEC section 5: the repository stores a SHA-256 per file, so a
mismatch means EMA published a new corpus version. That is reported, not raised.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx

from medleaf.ingest.pdf import detect_language, extract_lines
from medleaf.models import Lang, Medicine


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
) -> DownloadResult:
    """Download one PDF, verify its language when asked, and hash it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = client.get(url)
    response.raise_for_status()
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
