#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --nodelist=node01
#SBATCH --job-name=relayspec-score-math500
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec}"
: "${RESULT_DIR:?RESULT_DIR must point to a completed benchmark output directory}"
: "${RELAYSPEC_PYTHON:?RELAYSPEC_PYTHON must be set}"
QWEN_MATH_EVAL="${QWEN_MATH_EVAL:-$PROJECT_DIR/vendor/qwen2.5-math/evaluation}"
QWEN_SCORE_DEPS="${QWEN_SCORE_DEPS:-$PROJECT_DIR/vendor/qwen-score-deps}"

export PYTHONPATH="$PROJECT_DIR/src:$QWEN_SCORE_DEPS${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$PROJECT_DIR"
sha256sum scripts/aggregate_results.py src/relayspec/evaluation.py \
    src/relayspec/metrics.py > "$RESULT_DIR/scorer-source-sha256.txt"
git -C "$PROJECT_DIR/vendor/qwen2.5-math" rev-parse HEAD \
    > "$RESULT_DIR/qwen-math-eval-revision.txt"

EXTRA_ARGS=()
if [[ -n "${SUBSET_MANIFEST:-}" ]]; then
    EXTRA_ARGS+=(--subset-manifest "$SUBSET_MANIFEST")
fi
if [[ -n "${OUTPUT_PREFIX:-}" ]]; then
    EXTRA_ARGS+=(--output-prefix "$OUTPUT_PREFIX")
fi
"$RELAYSPEC_PYTHON" scripts/aggregate_results.py "$RESULT_DIR" \
    --qwen-math-eval "$QWEN_MATH_EVAL" \
    --bootstrap-samples 10000 \
    --seed 1729 \
    "${EXTRA_ARGS[@]}" \
    | tee "$RESULT_DIR/${OUTPUT_PREFIX:-benchmark}-scoring.log"
