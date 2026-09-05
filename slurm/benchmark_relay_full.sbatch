#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --nodelist=node01
#SBATCH --job-name=relayspec-math500-full
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail
module load u22/cuda/12.4

PROJECT_DIR="${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec}"
CONFIG="${CONFIG:-configs/benchmark_relay_qwen3_14b_math500_full_4gpu.yaml}"
OUTPUT_DIR="$PROJECT_DIR/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
: "${RELAYSPEC_PYTHON:?RELAYSPEC_PYTHON must be set}"
: "${RELAYSPEC_CACHE_DIR:?RELAYSPEC_CACHE_DIR must be set}"
: "${DFLASH_SOURCE:?DFLASH_SOURCE must be set}"

mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/logs"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export RELAYSPEC_OUTPUT="$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$DFLASH_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
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
cp scripts/benchmark_relay.py scripts/analyze_profile.py \
    scripts/check_development_gate.py "$OUTPUT_DIR/source-snapshot/scripts/"
cp src/relayspec/*.py "$OUTPUT_DIR/source-snapshot/src/relayspec/"
sha256sum scripts/benchmark_relay.py scripts/analyze_profile.py \
    scripts/check_development_gate.py src/relayspec/*.py "$CONFIG" \
    > "$OUTPUT_DIR/source-sha256.txt"
"$RELAYSPEC_PYTHON" scripts/check_allocation.py | tee "$OUTPUT_DIR/allocation.jsonl"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv -l 10 > "$OUTPUT_DIR/gpu.csv" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT

"$RELAYSPEC_PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node=4 \
    scripts/benchmark_relay.py --config "$CONFIG"

if [[ "$SLURM_JOB_NAME" == rs-val-* ]]; then
    "$RELAYSPEC_PYTHON" scripts/analyze_profile.py \
        --input "$OUTPUT_DIR"/benchmark-rank*.jsonl \
        --reference-method optimized_source_reuse \
        --candidate-method relay_p \
        --output-json "$OUTPUT_DIR/development-analysis.json" \
        --output-md "$OUTPUT_DIR/development-analysis.md" \
        --bootstrap-replicates 10000 \
        --seed 1729
    "$RELAYSPEC_PYTHON" scripts/check_development_gate.py \
        --analysis "$OUTPUT_DIR/development-analysis.json" \
        --expected-pairs 32 \
        --output "$OUTPUT_DIR/development-gate.json"
fi

kill "$GPU_MONITOR_PID" 2>/dev/null || true
trap - EXIT
