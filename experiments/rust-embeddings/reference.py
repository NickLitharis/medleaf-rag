from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import tokenizers as tokenizers_package
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from medleaf.retrieval.embed import E5Encoder, PASSAGE_PREFIX, QUERY_PREFIX, prefixed  # noqa: E402

MODEL_ID = "intfloat/multilingual-e5-base"
REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
ARTIFACT_FILES = (
    "config.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)
ALLOW_PATTERNS = [*ARTIFACT_FILES, "README.md"]
THREADS = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version_of(package: Any) -> str:
    return str(package.__version__)


def command_version(name: str) -> str:
    try:
        return subprocess.check_output([name, "--version"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def make_boundary_text(tokenizer: Any, target_count: int) -> tuple[str, int]:
    """Create a deterministic repeated-x body with exactly target_count raw tokens."""
    for word_count in range(max(1, target_count - 16), target_count + 16):
        text = " ".join(["x"] * word_count)
        actual = len(tokenizer(text, add_special_tokens=True)["input_ids"])
        if actual == target_count:
            return text, word_count
    raise RuntimeError(f"could not generate a {target_count}-token boundary sample")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reference-out", type=Path, required=True)
    parser.add_argument("--native-input", type=Path, required=True)
    args = parser.parse_args()

    cases = read_json(EXPERIMENT / "cases.json")
    if cases["model_id"] != MODEL_ID or cases["revision"] != REVISION:
        raise RuntimeError("cases.json model identity differs from the pinned probe identity")

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(THREADS)
    if torch.cuda.is_available() or torch.version.cuda is not None:
        raise RuntimeError("the reference probe must use a CPU-only PyTorch wheel")

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            cache_dir=str(cache_dir),
            allow_patterns=ALLOW_PATTERNS,
        )
    ).resolve()
    if snapshot.name != REVISION:
        raise RuntimeError(f"snapshot resolved to {snapshot.name}, not pinned revision {REVISION}")

    encoder = E5Encoder.load(
        model_id=MODEL_ID,
        revision=REVISION,
        device="cpu",
        batch_size=cases["batch_size"],
        max_seq_length=cases["max_seq_length"],
        cache_dir=cache_dir,
    )
    model: SentenceTransformer = encoder._model  # type: ignore[assignment]
    model.eval()
    transformer = model[0]
    tokenizer = transformer.tokenizer

    boundary_items: list[dict[str, Any]] = []
    boundary_repeats: dict[str, int] = {}
    for target in cases["boundary_token_counts"]:
        text, word_count = make_boundary_text(tokenizer, target)
        item_id = f"boundary_{target}"
        boundary_items.append({"id": item_id, "lang": "en", "text": text})
        boundary_repeats[item_id] = word_count

    queries = cases["queries"]
    passages = cases["passages"]
    items_by_id = {item["id"]: item for item in [*queries, *passages, *boundary_items]}
    raw_inputs = [
        {"id": item["id"], "text": item["text"]}
        for item in [*queries, *passages, *boundary_items]
    ]

    batches = [
        {"id": "queries_mixed_en_el", "task": "query", "item_ids": [item["id"] for item in queries]},
        {
            "id": "passages_en_el_1",
            "task": "passage",
            "item_ids": [item["id"] for item in passages[:4]],
        },
        {
            "id": "passages_en_el_2",
            "task": "passage",
            "item_ids": [item["id"] for item in passages[4:]],
        },
        {
            "id": "boundary_496_with_mixed_padding",
            "task": "passage",
            "item_ids": ["boundary_496", "p_en_adverse", "p_el_adverse"],
        },
        {
            "id": "boundary_512_513_with_mixed_padding",
            "task": "passage",
            "item_ids": ["boundary_512", "boundary_513", "p_en_adverse", "p_el_adverse"],
        },
    ]
    probe_batches: list[dict[str, Any]] = []
    for batch in batches:
        probe_batches.append(
            {
                "id": batch["id"],
                "task": batch["task"],
                "items": [
                    {"id": item_id, "text": items_by_id[item_id]["text"]}
                    for item_id in batch["item_ids"]
                ],
            }
        )

    pad_token_id = int(model[0].auto_model.config.pad_token_id)
    probe_input = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "max_seq_length": cases["max_seq_length"],
        "pad_token_id": pad_token_id,
        "raw_inputs": raw_inputs,
        "batches": probe_batches,
    }
    write_json(args.native_input, probe_input)

    raw_texts = [item["text"] for item in raw_inputs]
    raw_counts = encoder.count_tokens(raw_texts)
    raw_encoded = tokenizer(raw_texts, add_special_tokens=True)["input_ids"]
    raw_counts_by_id: dict[str, int] = {}
    raw_token_ids: dict[str, list[int]] = {}
    for item, count, ids in zip(raw_inputs, raw_counts, raw_encoded, strict=True):
        if count != len(ids):
            raise RuntimeError(f"E5Encoder.count_tokens disagrees with tokenizer for {item['id']}")
        raw_counts_by_id[item["id"]] = int(count)
        raw_token_ids[item["id"]] = [int(token_id) for token_id in ids]

    batch_token_ids: dict[str, dict[str, Any]] = {}
    vectors: dict[str, list[float]] = {}
    batch_vectors: dict[str, dict[str, list[float]]] = {}
    for batch in probe_batches:
        mode = batch["task"]
        prefix = QUERY_PREFIX if mode == "query" else PASSAGE_PREFIX
        forbidden = PASSAGE_PREFIX if mode == "query" else QUERY_PREFIX
        items = batch["items"]
        texts = [item["text"] for item in items]
        prefixed_texts = prefixed(texts, prefix, forbidden=forbidden)
        features = transformer.tokenize(prefixed_texts)
        input_ids = features["input_ids"].tolist()
        attention_mask = features["attention_mask"].tolist()
        batch_token_ids[batch["id"]] = {
            item["id"]: {
                "input_ids": [int(token_id) for token_id in ids],
                "attention_mask": [int(mask) for mask in masks],
            }
            for item, ids, masks in zip(items, input_ids, attention_mask, strict=True)
        }

        encoded = (
            encoder.encode_queries(texts)
            if mode == "query"
            else encoder.encode_passages(texts)
        )
        current_batch_vectors: dict[str, list[float]] = {}
        for item, vector in zip(items, encoded, strict=True):
            normalized = [float(value) for value in vector]
            current_batch_vectors[item["id"]] = normalized
            vectors.setdefault(item["id"], normalized)
        batch_vectors[batch["id"]] = current_batch_vectors

    file_hashes: dict[str, dict[str, Any]] = {}
    for relative in ARTIFACT_FILES:
        artifact = snapshot / relative
        if not artifact.is_file():
            raise FileNotFoundError(f"pinned model artifact is missing: {relative}")
        file_hashes[relative] = {"sha256": sha256(artifact), "size_bytes": artifact.stat().st_size}

    config = read_json(snapshot / "config.json")
    pooling = read_json(snapshot / "1_Pooling" / "config.json")
    metadata = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "snapshot_path": str(snapshot),
        "provenance": "Hugging Face snapshot_download at the exact model commit; safetensors loaded directly, no conversion",
        "files": file_hashes,
        "model_config": {
            "architectures": config["architectures"],
            "model_type": config["model_type"],
            "hidden_size": config["hidden_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "vocab_size": config["vocab_size"],
            "max_position_embeddings": config["max_position_embeddings"],
            "pad_token_id": config["pad_token_id"],
        },
        "sentence_transformers_pooling": pooling,
        "encoder_contract": {
            "query_prefix": QUERY_PREFIX,
            "passage_prefix": PASSAGE_PREFIX,
            "max_seq_length": cases["max_seq_length"],
            "normalize_embeddings": True,
            "pooling": "attention-mask-aware mean over all non-padding tokens, including special tokens",
            "model_mode": "eval",
        },
        "boundary_generator": {"token": "x", "word_counts": boundary_repeats},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "torch_cuda_available": torch.cuda.is_available(),
            "sentence_transformers": version_of(__import__("sentence_transformers")),
            "transformers": transformers.__version__,
            "tokenizers": tokenizers_package.__version__,
            "huggingface_hub": version_of(__import__("huggingface_hub")),
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "uv": command_version(os.environ.get("UV_BIN", "uv")),
            "rustc": command_version(os.environ.get("RUSTC_BIN", "rustc")),
            "cargo": command_version(os.environ.get("CARGO_BIN", "cargo")),
        },
        "sample_counts": {
            "queries": len(queries),
            "passages": len(passages),
            "boundary_inputs": len(boundary_items),
            "raw_inputs": len(raw_inputs),
            "inference_batches": len(probe_batches),
        },
    }
    reference = {
        "metadata": metadata,
        "raw_counts": raw_counts_by_id,
        "raw_token_ids": raw_token_ids,
        "batch_token_ids": batch_token_ids,
        "vectors": vectors,
        "batch_vectors": batch_vectors,
    }
    write_json(args.reference_out, reference)
    print(
        json.dumps(
            {
                "snapshot": str(snapshot),
                "raw_inputs": len(raw_inputs),
                "queries": len(queries),
                "passages": len(passages),
                "boundary_counts": {
                    item_id: raw_counts_by_id[item_id]
                    for item_id in boundary_repeats
                },
                "reference_out": str(args.reference_out),
                "native_input": str(args.native_input),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
