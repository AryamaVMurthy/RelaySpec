#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --nodelist=node01
#SBATCH --job-name=benchmark-relayspec-eagle3
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail
module load u22/cuda/12.4
PROJECT_DIR="${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec}"
CONFIG="${CONFIG:-configs/protocol_active/profile_eagle3_qwen3_8b_optimized_source_4gpu.yaml}"
OUTPUT_DIR="$PROJECT_DIR/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
: "${RELAYSPEC_PYTHON:?RELAYSPEC_PYTHON must be set}"
: "${RELAYSPEC_CACHE_DIR:?RELAYSPEC_CACHE_DIR must be set}"
: "${DEEPSPEC_SOURCE:?DEEPSPEC_SOURCE must be set}"
: "${DEEPSPEC_PYTHON_OVERLAY:?DEEPSPEC_PYTHON_OVERLAY must be set}"
mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/logs"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export RELAYSPEC_OUTPUT="$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$DEEPSPEC_PYTHON_OVERLAY:$DEEPSPEC_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT_DIR"
cp "$CONFIG" "$OUTPUT_DIR/config.yaml"
MANIFEST_PATH=$("$RELAYSPEC_PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    print(yaml.safe_load(stream)["benchmark"]["manifest_path"])
PY
)
cp "$MANIFEST_PATH" "$OUTPUT_DIR/eval-manifest.json"
sha256sum "$MANIFEST_PATH" > "$OUTPUT_DIR/eval-manifest.sha256"
mkdir -p "$OUTPUT_DIR/source-snapshot/scripts" "$OUTPUT_DIR/source-snapshot/src/relayspec"
cp scripts/benchmark_eagle3.py "$OUTPUT_DIR/source-snapshot/scripts/"
cp src/relayspec/*.py "$OUTPUT_DIR/source-snapshot/src/relayspec/"
sha256sum scripts/benchmark_eagle3.py src/relayspec/*.py "$CONFIG" \
    > "$OUTPUT_DIR/source-sha256.txt"
"$RELAYSPEC_PYTHON" scripts/check_allocation.py | tee "$OUTPUT_DIR/allocation.jsonl"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv -l 10 > "$OUTPUT_DIR/gpu.csv" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
"$RELAYSPEC_PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node=4 \
    scripts/benchmark_eagle3.py --config "$CONFIG"
kill "$GPU_MONITOR_PID" 2>/dev/null || true
trap - EXIT
