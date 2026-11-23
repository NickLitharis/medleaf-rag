"""Kaggle run 2 of 2: chunk the saved PDFs with the project's code and embed them.

Reads the PDFs from the medleaf-ema-pdfs kernel's output (no EMA download). Medicines left
out there (changed checksum) are skipped, with the dev questions that name them, and are
listed in manifest.json.
Output (/kaggle/working): vectors.npy, chunks.jsonl, query_vectors.npy, queries.jsonl,
manifest.json. See ../../PLAN.md.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/NickLitharis/medleaf-rag"
COMMIT = "bd805358da65d1e3c59e698ecd586e01eda6ae38"
MODEL, REVISION = "intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a"
WORK, OUT = Path("/kaggle/temp/medleaf"), Path("/kaggle/working")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pymupdf", "pydantic-settings",
                "pyyaml"], check=True)
subprocess.run(["git", "clone", "-q", REPO, str(WORK)], check=True)
subprocess.run(["git", "-C", str(WORK), "checkout", "-q", COMMIT], check=True)
sys.path.insert(0, str(WORK / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from medleaf.config import Settings  # noqa: E402
from medleaf.embed import E5Encoder  # noqa: E402
from medleaf.ingest.cache import chunk_budget_for, parse_corpus  # noqa: E402
from medleaf.ingest.store import load_medicines  # noqa: E402

settings = Settings()
medicines = load_medicines(settings.medicines_file)

# 1. Use the PDFs saved by the medleaf-ema-pdfs kernel; nothing is downloaded here.
found = list(Path("/kaggle/input").rglob("pdfs-manifest.json"))
assert found, "run the medleaf-ema-pdfs kernel first and attach it as an input"
pdf_manifest = json.loads(found[0].read_text(encoding="utf-8"))
shutil.copytree(found[0].parent / "pdfs", settings.raw_dir, dirs_exist_ok=True)
left_out = pdf_manifest["left_out"]
kept = [m for m in medicines if m.slug in set(pdf_manifest["kept"])]
print(f"kept {len(kept)} medicines, left out {len(left_out)}: {sorted(left_out)}")

# 2. Chunk with the project's parser, then embed exactly as `index build` does.
encoder = E5Encoder.load(model_id=MODEL, revision=REVISION,
                         device="cuda" if torch.cuda.is_available() else "cpu",
                         batch_size=64, max_seq_length=settings.embed_max_seq_length,
                         cache_dir=Path("/kaggle/temp/models"))
budget = chunk_budget_for(encoder, settings)
parsed, missing = parse_corpus(settings, kept, budget=budget, reuse_cache=False, progress=print)
assert not missing, f"missing documents: {missing}"

rows, blocks = [], []
started = time.perf_counter()
for result in parsed:
    blocks.append(np.asarray(encoder.encode_passages([s.text for s in result.sections]),
                             dtype=np.float32))
    rows += [{"chunk_id": s.chunk_id, "lang": s.lang.value} for s in result.sections]
vectors = np.concatenate(blocks)
embed_seconds = time.perf_counter() - started
assert len(vectors) == len(rows)

# 3. Dev questions that have gold sections and name only kept medicines. The held-out
#    test split is not touched.
kept_slugs = {m.slug for m in kept}
questions = [json.loads(line) for line in
             (WORK / "eval/datasets/retrieval-v1.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
dev = [q for q in questions if q["split"] == "dev"]
usable = [q for q in dev if q["gold_chunk_ids"] and set(q["medicines"]) <= kept_slugs]
query_vectors = np.asarray(encoder.encode_queries([q["question"] for q in usable]),
                           dtype=np.float32)

np.save(OUT / "vectors.npy", vectors)
np.save(OUT / "query_vectors.npy", query_vectors)
for name, items in (("chunks", rows), ("queries", usable)):
    (OUT / f"{name}.jsonl").write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
manifest = {
    "model": MODEL, "revision": REVISION, "repo_commit": COMMIT,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "chunks": len(rows), "dims": int(vectors.shape[1]), "embed_seconds": round(embed_seconds, 1),
    "max_norm_error": float(np.abs(np.linalg.norm(vectors, axis=1) - 1).max()),
    "medicines_kept": sorted(kept_slugs), "medicines_left_out": left_out,
    "dev_questions_total": len(dev), "dev_questions_used": len(usable),
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=1))
