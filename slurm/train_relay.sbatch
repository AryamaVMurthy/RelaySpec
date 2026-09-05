#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --job-name=train-relayspec
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail
module load u22/cuda/12.4

PROJECT_DIR="${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec}"
CONFIG="${CONFIG:-configs/train_relay_qwen3_4gpu.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_relay.py}"
OUTPUT_DIR="$PROJECT_DIR/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
: "${RELAYSPEC_PYTHON:?RELAYSPEC_PYTHON must be set}"
: "${RELAYSPEC_CACHE_DIR:?RELAYSPEC_CACHE_DIR must be set}"

mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/logs"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$RELAYSPEC_CACHE_DIR/hf-datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export RELAYSPEC_OUTPUT="$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${DFLASH_SOURCE:+:$DFLASH_SOURCE}${DEEPSPEC_SOURCE:+:$DEEPSPEC_SOURCE}${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"
cp "$CONFIG" "$OUTPUT_DIR/config.yaml"
mkdir -p "$OUTPUT_DIR/source-snapshot/scripts" "$OUTPUT_DIR/source-snapshot/src/relayspec"
cp "$TRAIN_SCRIPT" "$OUTPUT_DIR/source-snapshot/scripts/"
cp src/relayspec/*.py "$OUTPUT_DIR/source-snapshot/src/relayspec/"
sha256sum "$TRAIN_SCRIPT" src/relayspec/*.py "$CONFIG" \
    > "$OUTPUT_DIR/source-sha256.txt"
"$RELAYSPEC_PYTHON" scripts/check_allocation.py | tee "$OUTPUT_DIR/allocation.jsonl"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv -l 10 > "$OUTPUT_DIR/gpu.csv" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
"$RELAYSPEC_PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node=4 \
    "$TRAIN_SCRIPT" --config "$CONFIG"
kill "$GPU_MONITOR_PID" 2>/dev/null || true
trap - EXIT
