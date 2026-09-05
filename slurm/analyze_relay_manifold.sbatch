#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --nodelist=node01
#SBATCH --job-name=e4-p0-manifold
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail
module load u22/cuda/12.4
PROJECT_DIR="${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec}"
OUTPUT_DIR="$PROJECT_DIR/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
: "${RELAYSPEC_PYTHON:?RELAYSPEC_PYTHON must be set}"
: "${RELAYSPEC_CACHE_DIR:?RELAYSPEC_CACHE_DIR must be set}"
mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/logs"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT_DIR"
"$RELAYSPEC_PYTHON" scripts/analyze_relay_manifold.py \
    --checkpoint-a "$CHECKPOINT_A" \
    --target-a-id "$TARGET_A_ID" \
    --target-a-revision "$TARGET_A_REVISION" \
    --checkpoint-b "$CHECKPOINT_B" \
    --target-b-id "$TARGET_B_ID" \
    --target-b-revision "$TARGET_B_REVISION" \
    --text-manifest "$TEXT_MANIFEST" \
    --num-examples 32 \
    --cache-dir "$TRANSFORMERS_CACHE" \
    --output "$OUTPUT_DIR/manifold-probe.json"
