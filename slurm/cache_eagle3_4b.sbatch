#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --nodelist=node01
#SBATCH --job-name=relayspec-cache-eagle3-4b
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail
PROJECT_DIR="${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec}"
: "${RELAYSPEC_PYTHON:?RELAYSPEC_PYTHON must be set}"
: "${RELAYSPEC_CACHE_DIR:?RELAYSPEC_CACHE_DIR must be set}"
mkdir -p "$PROJECT_DIR/logs"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
cd "$PROJECT_DIR"
"$RELAYSPEC_PYTHON" scripts/cache_models.py \
    --cache-dir "$TRANSFORMERS_CACHE" \
    --model deepseek-ai/eagle3_qwen3_4b_ttt7 b0b90fd15d052217c226be5e46d468d8d129e0cd
