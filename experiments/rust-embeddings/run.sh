#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$EXPERIMENT_DIR/../.." && pwd)"
UV_BIN="${UV_BIN:-$(command -v uv 2>/dev/null || printf '%s' "$HOME/.local/bin/uv")}"
CARGO_BIN="${CARGO_BIN:-$(command -v cargo 2>/dev/null || printf '%s' "$HOME/.cargo/bin/cargo")}"
RUSTC_BIN="${RUSTC_BIN:-$(command -v rustc 2>/dev/null || printf '%s' "$HOME/.cargo/bin/rustc")}"
PYTHON_EXE="${PYTHON_EXE:-$(command -v python3.13 2>/dev/null || printf '%s' /usr/bin/python3.13)}"
VENV="${VENV:-/tmp/medleaf-embed-venv}"
HF_HOME="${HF_HOME:-/tmp/medleaf-rust-e5-cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
RUN_DIR="${RUN_DIR:-/tmp/medleaf-rust-embeddings-run}"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/medleaf-rust-embeddings-target}"

for executable in "$UV_BIN" "$CARGO_BIN" "$RUSTC_BIN" "$PYTHON_EXE"; do
  if [[ ! -x "$executable" ]]; then
    printf 'required executable is unavailable: %s\n' "$executable" >&2
    exit 1
  fi
done
export PATH="$(dirname -- "$CARGO_BIN"):$(dirname -- "$RUSTC_BIN"):$(dirname -- "$UV_BIN"):$PATH"
export UV_BIN CARGO_BIN RUSTC_BIN
export HF_HOME HF_HUB_CACHE
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export RAYON_NUM_THREADS=2
export CARGO_BUILD_JOBS=1
export CARGO_TARGET_DIR

mkdir -p "$RUN_DIR" "$HF_HUB_CACHE" "$CARGO_TARGET_DIR"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$UV_BIN" venv --python "$PYTHON_EXE" "$VENV"
fi
"$UV_BIN" pip install --python "$VENV/bin/python" -r "$EXPERIMENT_DIR/requirements.txt"

if [[ ! -f "$EXPERIMENT_DIR/Cargo.lock" ]]; then
  "$CARGO_BIN" generate-lockfile --manifest-path "$EXPERIMENT_DIR/Cargo.toml"
fi

"$VENV/bin/python" "$EXPERIMENT_DIR/reference.py" \
  --cache-dir "$HF_HUB_CACHE" \
  --reference-out "$RUN_DIR/python-reference.json" \
  --native-input "$RUN_DIR/probe-input.json"
MODEL_DIR="$("$VENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["metadata"]["snapshot_path"])' "$RUN_DIR/python-reference.json")"

"$CARGO_BIN" run --release --locked --manifest-path "$EXPERIMENT_DIR/Cargo.toml" -- \
  --model-dir "$MODEL_DIR" \
  --inputs "$RUN_DIR/probe-input.json" \
  --output "$RUN_DIR/native-output.json"

"$VENV/bin/python" "$EXPERIMENT_DIR/compare.py" \
  --reference "$RUN_DIR/python-reference.json" \
  --native "$RUN_DIR/native-output.json" \
  --native-input "$RUN_DIR/probe-input.json" \
  --out "$EXPERIMENT_DIR/results.json"
