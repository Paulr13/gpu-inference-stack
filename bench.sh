#!/usr/bin/env bash
# bench.sh — llama-bench sweep: models × context lengths × KV dtype → CSV.
#
# usage: ./bench.sh <models-dir> [out.csv]
# env:   LLAMA_BENCH  path to llama-bench (default: llama-bench on PATH)
#        CONTEXTS     context sizes to sweep   (default: "2048 8192 32768")
#        KV_TYPES     kv dtypes to sweep       (default: "f16 q8_0")
set -euo pipefail

MODELS_DIR="${1:?usage: bench.sh <models-dir> [out.csv]}"
OUT="${2:-bench-results.csv}"
LB="${LLAMA_BENCH:-llama-bench}"
CONTEXTS="${CONTEXTS:-2048 8192 32768}"
KV_TYPES="${KV_TYPES:-f16 q8_0}"

command -v "$LB" >/dev/null || { echo "llama-bench not found (set LLAMA_BENCH=...)" >&2; exit 1; }

echo "=== sweep: $(ls "$MODELS_DIR"/*.gguf 2>/dev/null | wc -l) model(s), ctx: $CONTEXTS, kv: $KV_TYPES ===" >&2

for m in "$MODELS_DIR"/*.gguf; do
  for kv in $KV_TYPES; do
    # llama-bench emits its own CSV header on each run — dedupe on import if needed
    "$LB" -m "$m" -p $CONTEXTS -n 128 -ctk "$kv" -ctv "$kv" -fa 1 -o csv >> "$OUT"
  done
done

echo "wrote $OUT" >&2