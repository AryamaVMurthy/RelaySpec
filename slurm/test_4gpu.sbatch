#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --job-name=relayspec-tests
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

module load u22/cuda/12.4

PROJECT_DIR="${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec}"
OUTPUT_DIR="$PROJECT_DIR/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
: "${RELAYSPEC_PYTHON:?RELAYSPEC_PYTHON must name a verified GPU Python environment}"
: "${RELAYSPEC_CACHE_DIR:?RELAYSPEC_CACHE_DIR must name the persistent model cache}"
: "${DFLASH_SOURCE:?DFLASH_SOURCE must name the pinned DFlash checkout}"

mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/logs"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$PROJECT_DIR/src:$DFLASH_SOURCE${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_DIR"
"$RELAYSPEC_PYTHON" scripts/check_allocation.py | tee "$OUTPUT_DIR/allocation.jsonl"
"$RELAYSPEC_PYTHON" -m pytest -q | tee "$OUTPUT_DIR/pytest.txt"
