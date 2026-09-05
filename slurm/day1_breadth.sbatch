#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --nodelist=node07
#SBATCH --job-name=rs-day1-pilot
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module load u22/cuda/12.4
: "${CONFIG:?Set CONFIG}"
: "${FAMILY:?Set FAMILY}"
cd "$HOME/relayspec-day1-20260905"
export RELAYSPEC_PYTHON="$HOME/.venvs/factorspec-20260826/bin/python"
export RELAYSPEC_CACHE_DIR="/scratch/$USER/factorspec-runtime-20260826"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export DFLASH_SOURCE="$HOME/factorspec/vendor/dflash"
export DEEPSPEC_SOURCE="$HOME/relayspec/vendor/deepspec"
export DEEPSPEC_PYTHON_OVERLAY="$RELAYSPEC_CACHE_DIR/deepspec-python-overlay-node07"
export PYTHONPATH="$PWD/src:$DFLASH_SOURCE"
if [[ "$FAMILY" == eagle3 ]]; then
    export PYTHONPATH="$PWD/src:$DEEPSPEC_PYTHON_OVERLAY:$DEEPSPEC_SOURCE"
fi
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export RELAYSPEC_OUTPUT="$PWD/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RELAYSPEC_OUTPUT"
cp "$CONFIG" "$RELAYSPEC_OUTPUT/config.yaml"
sha256sum scripts/trace_decoder_pilot.py scripts/benchmark_relay.py scripts/benchmark_eagle3.py src/relayspec/*.py "$CONFIG" > "$RELAYSPEC_OUTPUT/source-sha256.txt"
"$RELAYSPEC_PYTHON" scripts/check_allocation.py > "$RELAYSPEC_OUTPUT/allocation.jsonl"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used --format=csv -l 10 > "$RELAYSPEC_OUTPUT/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
timeout --signal=TERM --kill-after=10s "${RUN_TIMEOUT_SECONDS:-14000}s" "$RELAYSPEC_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/benchmark_eagle3.py --config "$CONFIG"
"$RELAYSPEC_PYTHON" scripts/check_run_complete.py --config "$CONFIG" --output "$RELAYSPEC_OUTPUT"
