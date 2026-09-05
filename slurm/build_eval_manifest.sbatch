#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --nodelist=node01
#SBATCH --job-name=relayspec-eval-manifest
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec}"
: "${RELAYSPEC_PYTHON:?RELAYSPEC_PYTHON must be set}"
: "${RELAYSPEC_CACHE_DIR:?RELAYSPEC_CACHE_DIR must be set}"
mkdir -p "$PROJECT_DIR/logs" "$RELAYSPEC_CACHE_DIR/hf-datasets"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$RELAYSPEC_CACHE_DIR/hf-datasets"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
OUTPUT_PATH="${OUTPUT_PATH:-configs/eval_manifest_full.json}"

cd "$PROJECT_DIR"
"$RELAYSPEC_PYTHON" scripts/build_eval_manifests.py \
    --output "$OUTPUT_PATH" \
    --cache-dir "$HF_DATASETS_CACHE" \
    --seed 42
sha256sum "$OUTPUT_PATH" > "${OUTPUT_PATH%.json}.sha256"
