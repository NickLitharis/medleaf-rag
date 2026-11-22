from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT = Path(__file__).resolve().parent
# These gates are fixed before observing either backend's outputs.
VECTOR_TOLERANCES = {
    "max_abs_component_difference": 5e-4,
    "minimum_cosine_similarity": 0.99999,
    "max_l2_norm_difference": 1e-5,
    "max_norm_error_from_one": 1e-5,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids: list[int]) -> str:
    packed = struct.pack(f"<{len(ids)}I", *ids)
    return sha256_bytes(packed)


def compare_vectors(reference: list[float], native: list[float]) -> dict[str, Any]:
    expected = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(native, dtype=np.float64)
    if expected.shape != actual.shape:
        return {"dimension_match": False, "reference_dimension": int(expected.size), "native_dimension": int(actual.size), "pass": False}
    difference = actual - expected
    reference_norm = float(np.linalg.norm(expected))
    native_norm = float(np.linalg.norm(actual))
    cosine = float(np.dot(expected, actual) / (reference_norm * native_norm))
    max_abs = float(np.max(np.abs(difference)))
    norm_difference = abs(native_norm - reference_norm)
    norm_error = max(abs(reference_norm - 1.0), abs(native_norm - 1.0))
    passed = (
        max_abs <= VECTOR_TOLERANCES["max_abs_component_difference"]
        and cosine >= VECTOR_TOLERANCES["minimum_cosine_similarity"]
        and norm_difference <= VECTOR_TOLERANCES["max_l2_norm_difference"]
        and norm_error <= VECTOR_TOLERANCES["max_norm_error_from_one"]
    )
    return {
        "dimension_match": True,
        "dimension": int(expected.size),
        "max_abs_component_difference": max_abs,
        "cosine_similarity": cosine,
        "reference_l2_norm": reference_norm,
        "native_l2_norm": native_norm,
        "l2_norm_difference": norm_difference,
        "max_norm_error_from_one": norm_error,
        "pass": passed,
    }


def rank_passages(query_vector: list[float], passages: dict[str, list[float]]) -> list[str]:
    query = np.asarray(query_vector, dtype=np.float64)
    scores = []
    for passage_id, vector in passages.items():
        score = float(np.dot(query, np.asarray(vector, dtype=np.float64)))
        scores.append((passage_id, score))
    return [passage_id for passage_id, _ in sorted(scores, key=lambda pair: (-pair[1], pair[0]))]


def kendall_tau(left: list[str], right: list[str]) -> float:
    right_position = {item: index for index, item in enumerate(right)}
    concordant = 0
    discordant = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            if right_position[left[first]] < right_position[left[second]]:
                concordant += 1
            else:
                discordant += 1
    pairs = concordant + discordant
    return 1.0 if pairs == 0 else (concordant - discordant) / pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--native-input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reference = read_json(args.reference)
    native = read_json(args.native)
    native_input = read_json(args.native_input)
    metadata = reference["metadata"]
    cases = read_json(EXPERIMENT / "cases.json")
    files = metadata["files"]

    raw_checks: dict[str, dict[str, Any]] = {}
    raw_ids_equal = True
    for item in native_input["raw_inputs"]:
        item_id = item["id"]
        python_ids = reference["raw_token_ids"][item_id]
        native_ids = native["raw_token_ids"][item_id]
        python_count = int(reference["raw_counts"][item_id])
        native_count = int(native["raw_counts"][item_id])
        exact = python_ids == native_ids
        count_exact = python_count == native_count == len(python_ids) == len(native_ids)
        raw_ids_equal = raw_ids_equal and exact and count_exact
        raw_checks[item_id] = {
            "token_count_python": python_count,
            "token_count_native": native_count,
            "token_count_equal": count_exact,
            "token_ids_equal": exact,
            "python_token_ids_sha256_u32le": ids_sha256(python_ids),
            "native_token_ids_sha256_u32le": ids_sha256(native_ids),
        }

    boundary_ids = {f"boundary_{count}" for count in cases["boundary_token_counts"]}
    boundary_evidence: dict[str, Any] = {}
    for item_id in sorted(boundary_ids, key=lambda value: int(value.rsplit("_", 1)[1])):
        python_ids = reference["raw_token_ids"][item_id]
        native_ids = native["raw_token_ids"][item_id]
        boundary_evidence[item_id] = {
            "count_python": len(python_ids),
            "count_native": len(native_ids),
            "python_unprefixed_token_ids": python_ids,
            "native_unprefixed_token_ids": native_ids,
            "unprefixed_token_ids_equal": python_ids == native_ids,
        }

    batch_checks: dict[str, Any] = {}
    all_batch_ids_equal = True
    batch_item_count = 0
    for batch in native_input["batches"]:
        batch_id = batch["id"]
        python_batch = reference["batch_token_ids"][batch_id]
        native_batch = native["batch_token_ids"][batch_id]
        item_checks: dict[str, Any] = {}
        widths = set()
        for item in batch["items"]:
            item_id = item["id"]
            python_ids = python_batch[item_id]["input_ids"]
            native_ids = native_batch[item_id]["input_ids"]
            python_mask = python_batch[item_id]["attention_mask"]
            native_mask = native_batch[item_id]["attention_mask"]
            ids_equal = python_ids == native_ids
            masks_equal = python_mask == native_mask
            all_batch_ids_equal = all_batch_ids_equal and ids_equal and masks_equal
            batch_item_count += 1
            widths.add(len(python_ids))
            item_checks[item_id] = {
                "input_ids_equal": ids_equal,
                "attention_mask_equal": masks_equal,
                "token_count_before_padding": sum(python_mask),
                "padded_width": len(python_ids),
                "python_input_ids_sha256_u32le": ids_sha256(python_ids),
                "native_input_ids_sha256_u32le": ids_sha256(native_ids),
            }
            if item_id in boundary_ids:
                boundary_evidence[item_id]["model_batch"] = batch_id
                boundary_evidence[item_id]["python_model_input_ids"] = python_ids
                boundary_evidence[item_id]["native_model_input_ids"] = native_ids
                boundary_evidence[item_id]["python_model_attention_mask"] = python_mask
                boundary_evidence[item_id]["native_model_attention_mask"] = native_mask
                boundary_evidence[item_id]["model_input_token_ids_equal"] = ids_equal
                boundary_evidence[item_id]["model_attention_mask_equal"] = masks_equal
        langs = sorted({
            item.get("lang", "generated")
            for item in [
                *cases["queries"],
                *cases["passages"],
                *[{"id": f"boundary_{count}", "lang": "en"} for count in cases["boundary_token_counts"]],
            ]
            if item["id"] in {row["id"] for row in batch["items"]}
        })
        batch_checks[batch_id] = {
            "task": batch["task"],
            "item_count": len(batch["items"]),
            "languages": langs,
            "padded_widths": sorted(widths),
            "items": item_checks,
            "exact_input_ids_and_masks": all(
                item["input_ids_equal"] and item["attention_mask_equal"]
                for item in item_checks.values()
            ),
        }

    vector_checks: dict[str, Any] = {}
    for item_id, python_vector in reference["vectors"].items():
        vector_checks[item_id] = compare_vectors(python_vector, native["vectors"][item_id])
    vector_values = list(vector_checks.values())
    vector_pass = bool(vector_values) and all(item["pass"] for item in vector_values)
    max_abs = max(item["max_abs_component_difference"] for item in vector_values)
    min_cosine = min(item["cosine_similarity"] for item in vector_values)
    max_norm_delta = max(item["l2_norm_difference"] for item in vector_values)
    max_norm_error = max(item["max_norm_error_from_one"] for item in vector_values)

    query_items = cases["queries"]
    passage_ids = [passage["id"] for passage in cases["passages"]]
    python_passages = {item_id: reference["vectors"][item_id] for item_id in passage_ids}
    native_passages = {item_id: native["vectors"][item_id] for item_id in passage_ids}
    rankings = []
    for query in query_items:
        python_order = rank_passages(reference["vectors"][query["id"]], python_passages)
        native_order = rank_passages(native["vectors"][query["id"]], native_passages)
        rankings.append(
            {
                "query_id": query["id"],
                "reference_order": python_order,
                "native_order": native_order,
                "full_order_equal": python_order == native_order,
                "top1_equal": python_order[:1] == native_order[:1],
                "top3_equal": set(python_order[:3]) == set(native_order[:3]),
                "kendall_tau": kendall_tau(python_order, native_order),
            }
        )

    lock_path = EXPERIMENT / "Cargo.lock"
    lock_data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    selected_crates = {"candle-core", "candle-nn", "candle-transformers", "tokenizers"}
    crate_versions = {
        name: sorted(
            {
                package["version"]
                for package in lock_data["package"]
                if package["name"] == name
            }
        )
        for name in sorted(selected_crates)
    }
    probe_package = next(
        package for package in lock_data["package"] if package["name"] == "rust-e5-probe"
    )
    direct_tokenizer_version = next(
        dependency.split(" ", 1)[1]
        for dependency in probe_package["dependencies"]
        if dependency.startswith("tokenizers ")
    )
    source_paths = [
        EXPERIMENT / "Cargo.toml",
        EXPERIMENT / "Cargo.lock",
        EXPERIMENT / "requirements.txt",
        EXPERIMENT / "cases.json",
        EXPERIMENT / "src" / "main.rs",
        EXPERIMENT / "reference.py",
        EXPERIMENT / "compare.py",
        EXPERIMENT / "run.sh",
    ]
    source_hashes = {
        str(path.relative_to(EXPERIMENT)): sha256_file(path)
        for path in source_paths
        if path.is_file()
    }
    metadata["native_runtime"] = {
        "backend": native["backend"],
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "cargo": subprocess.check_output(["cargo", "--version"], text=True).strip(),
        "candle_crate_versions": crate_versions,
        "cargo_lock_sha256": sha256_file(lock_path),
        "rayon_num_threads": int(__import__("os").environ.get("RAYON_NUM_THREADS", "0")),
        "cargo_build_jobs": int(__import__("os").environ.get("CARGO_BUILD_JOBS", "0")),
        "native_inference_device": "CPU",
        "native_inference_dtype": "f32",
        "probe_direct_tokenizers_version": direct_tokenizer_version,
    }
    metadata["probe_source_sha256"] = source_hashes
    metadata["runtime"]["uv"] = subprocess.check_output(["uv", "--version"], text=True).strip()

    result = {
        "experiment": "native Rust Candle inference feasibility against the current Python E5Encoder",
        "model": metadata,
        "acceptance": {
            "tolerances_predeclared_in_source": str((EXPERIMENT / "compare.py").relative_to(EXPERIMENT)),
            "vector_tolerances": VECTOR_TOLERANCES,
            "raw_token_counts_exact": all(item["token_count_equal"] for item in raw_checks.values()),
            "raw_token_ids_exact": raw_ids_equal,
            "padded_batch_token_ids_and_masks_exact": all_batch_ids_equal,
            "all_vector_metrics_pass": vector_pass,
            "pass": raw_ids_equal and all_batch_ids_equal and vector_pass,
        },
        "token_parity": {
            "raw_inputs": raw_checks,
            "exact_boundary_token_ids": boundary_evidence,
            "batches": batch_checks,
            "counts": {
                "raw_inputs": len(raw_checks),
                "raw_token_sequences_equal": sum(item["token_ids_equal"] for item in raw_checks.values()),
                "raw_token_count_equal": sum(item["token_count_equal"] for item in raw_checks.values()),
                "padded_batch_items": batch_item_count,
                "padded_batch_ids_and_masks_equal": sum(
                    item["input_ids_equal"] and item["attention_mask_equal"]
                    for batch in batch_checks.values()
                    for item in batch["items"].values()
                ),
            },
        },
        "vector_comparison": {
            "sample_count": len(vector_checks),
            "max_abs_component_difference": max_abs,
            "minimum_cosine_similarity": min_cosine,
            "max_l2_norm_difference": max_norm_delta,
            "max_norm_error_from_one": max_norm_error,
            "python_norm_range": [
                min(item["reference_l2_norm"] for item in vector_values),
                max(item["reference_l2_norm"] for item in vector_values),
            ],
            "native_norm_range": [
                min(item["native_l2_norm"] for item in vector_values),
                max(item["native_l2_norm"] for item in vector_values),
            ],
            "per_input": vector_checks,
        },
        "nearest_neighbor_ranking_sample_only": {
            "queries": len(query_items),
            "passages": len(passage_ids),
            "top1_agreement_count": sum(item["top1_equal"] for item in rankings),
            "full_order_agreement_count": sum(item["full_order_equal"] for item in rankings),
            "mean_kendall_tau": float(np.mean([item["kendall_tau"] for item in rankings])),
            "per_query": rankings,
            "scope_note": "Small fixed embedding sample only; this is not full corpus, Qdrant, or retrieval parity evidence.",
        },
        "sample_counts": {
            **metadata["sample_counts"],
            "raw_token_sequences_compared": len(raw_checks),
            "padded_inference_rows_compared": batch_item_count,
            "vectors_compared": len(vector_checks),
            "queries_in_ranking": len(query_items),
            "passages_in_ranking": len(passage_ids),
        },
    }
    write_json(args.out, result)
    print(
        json.dumps(
            {
                "result": str(args.out),
                "pass": result["acceptance"]["pass"],
                "token_counts_equal": result["acceptance"]["raw_token_counts_exact"],
                "raw_token_ids_equal": result["acceptance"]["raw_token_ids_exact"],
                "padded_batch_ids_equal": result["acceptance"]["padded_batch_token_ids_and_masks_exact"],
                "vectors_pass": result["acceptance"]["all_vector_metrics_pass"],
                "max_abs": max_abs,
                "min_cosine": min_cosine,
                "max_norm_delta": max_norm_delta,
                "top1_agreement": result["nearest_neighbor_ranking_sample_only"]["top1_agreement_count"],
                "full_ranking_agreement": result["nearest_neighbor_ranking_sample_only"]["full_order_agreement_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
