#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --nodelist=node01
#SBATCH --job-name=relayspec-score-evalplus
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec}"
: "${RESULT_DIR:?RESULT_DIR must point to a completed benchmark output directory}"
: "${RELAYSPEC_PYTHON:?RELAYSPEC_PYTHON must be set}"
EVALPLUS_SOURCE="${EVALPLUS_SOURCE:-$PROJECT_DIR/vendor/evalplus}"
EVALPLUS_DEPS="${EVALPLUS_DEPS:-$PROJECT_DIR/vendor/evalplus-deps}"

export PYTHONPATH="$PROJECT_DIR/src:$EVALPLUS_SOURCE:$EVALPLUS_DEPS${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${RELAYSPEC_CACHE_DIR:-/scratch/aryama.murthy/factorspec-runtime-20260826}/huggingface"

cd "$PROJECT_DIR"
"$RELAYSPEC_PYTHON" scripts/export_evalplus.py "$RESULT_DIR"
git -C "$EVALPLUS_SOURCE" rev-parse HEAD \
    > "$RESULT_DIR/evalplus/evalplus-revision.txt"

"$RELAYSPEC_PYTHON" - "$RESULT_DIR/evalplus/manifest.json" \
    > "$RESULT_DIR/evalplus/files.txt" <<'PY'
import json
import sys

for row in json.load(open(sys.argv[1], encoding="utf-8"))["files"]:
    print(row["benchmark"], row["method"], row["path"], sep="\t")
PY

while IFS=$'\t' read -r benchmark method samples; do
    "$RELAYSPEC_PYTHON" -m evalplus.sanitize --samples "$samples"
    sanitized="${samples%.jsonl}-sanitized.jsonl"
    full_result="${sanitized%.jsonl}-full.eval_results.json"
    "$RELAYSPEC_PYTHON" -m evalplus.evaluate "$benchmark" \
        --samples="$sanitized" \
        --parallel="$SLURM_CPUS_PER_TASK" \
        --i_just_wanna_run=True \
        --output_file="$full_result" \
        2>&1 | tee "$RESULT_DIR/evalplus/${benchmark}-${method}.log"
done < "$RESULT_DIR/evalplus/files.txt"

"$RELAYSPEC_PYTHON" scripts/score_mbpp.py "$RESULT_DIR" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --timeout-seconds 5 \
    | tee "$RESULT_DIR/evalplus/mbpp-base.log"

"$RELAYSPEC_PYTHON" scripts/summarize_evalplus.py "$RESULT_DIR" \
    | tee "$RESULT_DIR/evalplus/summary.log"
