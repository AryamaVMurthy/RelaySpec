#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --time=00:10:00
#SBATCH --nodelist=node07
#SBATCH --job-name=rs-pard2-pilot
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module load u22/cuda/12.4
cd "${RELAYSPEC_PROJECT_DIR:?}"
: "${PARD_SETUP_GATE:?}" "${PARD2_CONFIG:?}"
export RELAYSPEC_CACHE_DIR="/scratch/$USER/factorspec-runtime-20260826"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface" TRANSFORMERS_CACHE="$RELAYSPEC_CACHE_DIR/huggingface/hub"
export RELAYSPEC_PYTHON="$HOME/.venvs/factorspec-20260826/bin/python"
export PYTHONPATH="$RELAYSPEC_CACHE_DIR/pard-python-overlay-27823:$PWD/src"
export PARD_SOURCE="$PWD/vendor/pard"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export RELAYSPEC_OUTPUT="$PWD/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RELAYSPEC_OUTPUT"
cp SOURCE_COMMIT "$RELAYSPEC_OUTPUT/source-commit.txt"
cp "$PARD2_CONFIG" "$RELAYSPEC_OUTPUT/config.json"
cp "$PARD_SETUP_GATE" "$RELAYSPEC_OUTPUT/setup-gate.json"
tar -czf "$RELAYSPEC_OUTPUT/source-snapshot.tar.gz" src scripts configs slurm vendor/pard SOURCE_COMMIT
"$RELAYSPEC_PYTHON" scripts/check_allocation.py > "$RELAYSPEC_OUTPUT/allocation.jsonl"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used --format=csv -l 10 > "$RELAYSPEC_OUTPUT/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
timeout --signal=TERM --kill-after=10s "${PARD2_TIMEOUT:-540}s" "$RELAYSPEC_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/benchmark_pard2.py --config "$PARD2_CONFIG"
