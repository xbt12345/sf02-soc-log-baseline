#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TRAIN_FILE=${1:-/root/work/sf02_data/train.parquet}
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
conda run --no-capture-output -n cm-model python "$BUNDLE_DIR/verify_bundle.py"
conda run --no-capture-output -n cm-model python -m unittest discover -s "$BUNDLE_DIR/training" -p 'test*32*.py'
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR="$BUNDLE_DIR/runs/round_$STAMP"
mkdir -p "$BUNDLE_DIR/run_logs"
conda run --no-capture-output -n cm-model python -u "$BUNDLE_DIR/training/run_v32_train.py" --train "$TRAIN_FILE" --run-dir "$BUNDLE_DIR/prepared" --output-dir "$OUTPUT_DIR" 2>&1 | tee "$BUNDLE_DIR/run_logs/round_$STAMP.log"
conda run --no-capture-output -n cm-model python -u "$BUNDLE_DIR/training/review_v32_round.py" --run-dir "$BUNDLE_DIR/prepared" --experiment-dir "$OUTPUT_DIR" 2>&1 | tee "$BUNDLE_DIR/run_logs/review_$STAMP.log"
conda run --no-capture-output -n cm-model python -m zipfile -c "$OUTPUT_DIR.zip" "$OUTPUT_DIR"
printf '\nResult folder: %s\nReturn this archive for review: %s.zip\n' "$OUTPUT_DIR" "$OUTPUT_DIR"
