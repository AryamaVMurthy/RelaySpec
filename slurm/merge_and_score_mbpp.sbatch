#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodelist=node01
#SBATCH --job-name=relayspec-merge-score-mbpp
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec}"
: "${FIRST_DIR:?FIRST_DIR must point to the first completed MBPP shard}"
: "${REMAINDER_DIR:?REMAINDER_DIR must point to the disjoint remainder shard}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}}"
: "${CONFIG:?CONFIG must describe the full 378-task run}"
: "${MANIFEST:?MANIFEST must contain all 378 pinned EvalPlus MBPP tasks}"
: "${RELAYSPEC_PYTHON:?RELAYSPEC_PYTHON must be set}"

export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_DIR"
"$RELAYSPEC_PYTHON" scripts/merge_benchmark_runs.py \
    --first-dir "$FIRST_DIR" \
    --remainder-dir "$REMAINDER_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --world-size 4

"$RELAYSPEC_PYTHON" scripts/aggregate_results.py "$OUTPUT_DIR" \
    --bootstrap-samples 10000 \
    --seed 1729 \
    | tee "$OUTPUT_DIR/scoring.log"

RESULT_DIR="$OUTPUT_DIR" bash slurm/score_evalplus.sbatch
