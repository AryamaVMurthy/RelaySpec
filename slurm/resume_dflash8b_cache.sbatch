#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --nodelist=node01
#SBATCH --job-name=resume-dflash8b
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec}"
: "${RELAYSPEC_PYTHON:?RELAYSPEC_PYTHON must be set}"
: "${RELAYSPEC_CACHE_DIR:?RELAYSPEC_CACHE_DIR must be set}"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600

cd "$PROJECT_DIR"
"$RELAYSPEC_PYTHON" scripts/resume_hf_shards.py \
    --repo z-lab/Qwen3-8B-DFlash-b16 \
    --revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
    --cache-dir "$TRANSFORMERS_CACHE" \
    --filename model.safetensors \
    --workers 1 \
    --connections-per-shard 12
"$RELAYSPEC_PYTHON" scripts/cache_models.py \
    --cache-dir "$TRANSFORMERS_CACHE" \
    --model z-lab/Qwen3-8B-DFlash-b16 9b41424b7109f9c5413454f481b09a82b85333f4
