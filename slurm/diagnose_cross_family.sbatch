#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --job-name=diagnose-cross-family
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail
module load u22/cuda/12.4
PROJECT_DIR="${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec}"
OUTPUT_DIR="$PROJECT_DIR/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
: "${RELAYSPEC_PYTHON:?RELAYSPEC_PYTHON must be set}"
: "${RELAYSPEC_CACHE_DIR:?RELAYSPEC_CACHE_DIR must be set}"
: "${DFLASH_SOURCE:?DFLASH_SOURCE must be set}"
mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/logs"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$PROJECT_DIR/src:$DFLASH_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT_DIR"
"$RELAYSPEC_PYTHON" scripts/diagnose_cross_family_divergence.py \
    --target-id unsloth/Llama-3.1-8B-Instruct \
    --target-revision 4699cc75b550f9c6f3173fb80f4703b62d946aa5 \
    --proposer-id z-lab/Qwen3-4B-DFlash-b16 \
    --proposer-revision b74e3a329c4d963783143b1e970d95b002be72bd \
    --proposer-source-commit 94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756 \
    --source-id Qwen/Qwen3-4B \
    --source-revision 1cfa9a7208912126459214e8b04321603b3df60c \
    --relay-checkpoint "${RELAY_CHECKPOINT:-/scratch/aryama.murthy/factorspec-runtime-20260826/relayspec/checkpoints/x1-qwen-dflash4b-relay-llama31-8b/relay.pt}" \
    --dflash-source "$DFLASH_SOURCE" \
    --cache-dir "$TRANSFORMERS_CACHE" \
    --prompt "${DIAGNOSE_PROMPT:-What is the derivative of x squared plus three x?}" \
    --max-new-tokens 64 \
    --block-size 16 \
    --output "$OUTPUT_DIR/diagnosis.json"
