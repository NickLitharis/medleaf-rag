"""Kaggle run 1 of 2: download and verify the pinned EMA PDFs, once.

Output (/kaggle/working): pdfs/<slug>/<slug>_<lang>.pdf for every medicine whose two PDFs
both match the checksum pinned in data/medicines.yaml, plus pdfs-manifest.json listing the
kept medicines and the ones left out (with the checksum EMA serves now). The embedding
kernel reads this as an input, so it never contacts EMA. Re-run this kernel only when the
corpus definition changes. See ../../PLAN.md.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/NickLitharis/medleaf-rag"
COMMIT = "bd805358da65d1e3c59e698ecd586e01eda6ae38"  # download code is unchanged since this commit
WORK, OUT = Path("/kaggle/temp/medleaf"), Path("/kaggle/working")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydantic-settings", "pyyaml",
                "httpx", "pymupdf"], check=True)
subprocess.run(["git", "clone", "-q", REPO, str(WORK)], check=True)
subprocess.run(["git", "-C", str(WORK), "checkout", "-q", COMMIT], check=True)
sys.path.insert(0, str(WORK / "src"))

import shutil  # noqa: E402

import httpx  # noqa: E402

from medleaf.config import Settings  # noqa: E402
from medleaf.ingest.download import download_medicine  # noqa: E402
from medleaf.ingest.store import load_medicines  # noqa: E402

settings = Settings()
medicines = load_medicines(settings.medicines_file)
kept, left_out = [], {}
with httpx.Client(headers={"User-Agent": settings.user_agent}, follow_redirects=True,
                  timeout=settings.request_timeout_s) as client:
    for medicine in medicines:
        try:
            results = download_medicine(client, medicine, settings.raw_dir)
        except Exception as error:
            left_out[medicine.slug] = {"reason": f"download failed: {error}"}
            continue
        bad = [r for r in results if r.changed or r.status != "verified"]
        if bad:
            left_out[medicine.slug] = {
                "reason": "checksum changed or language mismatch",
                "served": {r.lang.value: r.sha256 for r in results},
                "pinned": {r.lang.value: r.expected_sha256 for r in results},
            }
            continue
        for r in results:
            target = OUT / "pdfs" / medicine.slug / r.path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(r.path, target)
        kept.append(medicine.slug)
        time.sleep(0.2)  # be gentle with EMA

manifest = {"repo_commit": COMMIT, "kept": kept, "left_out": left_out}
(OUT / "pdfs-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
print(f"kept {len(kept)} medicines, left out {len(left_out)}: {sorted(left_out)}")
