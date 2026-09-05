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
#SBATCH --job-name=rs-cached-fit-pilot
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module load u22/cuda/12.4
cd "${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec-cached-scaling-20260905}"
export RELAYSPEC_PYTHON="$HOME/.venvs/factorspec-20260826/bin/python"
export RELAYSPEC_CACHE_DIR="/scratch/$USER/factorspec-runtime-20260826"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export DFLASH_SOURCE="$HOME/factorspec/vendor/dflash"
export DEEPSPEC_SOURCE="$HOME/relayspec/vendor/deepspec"
export DEEPSPEC_PYTHON_OVERLAY="$RELAYSPEC_CACHE_DIR/deepspec-python-overlay-node07"
family="$("$RELAYSPEC_PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["proposer"]["family"])' "${CACHE_TRAINING_CONFIG:-configs/protocol_active/train_dflash_qwen3_8b_relative_4gpu.yaml}")"
case "$family" in
  dflash) export PYTHONPATH="$PWD/src:$DFLASH_SOURCE" ;;
  eagle3)
    test -d "$DEEPSPEC_PYTHON_OVERLAY/transformers"
    export PYTHONPATH="$PWD/src:$DEEPSPEC_PYTHON_OVERLAY:$DEEPSPEC_SOURCE"
    ;;
  *) echo "Unsupported proposer family: $family" >&2; exit 2 ;;
esac
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export RELAYSPEC_OUTPUT="$PWD/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
export RELAYSPEC_FEATURE_CACHE="$RELAYSPEC_CACHE_DIR/relayspec/feature-caches/pilot-${SLURM_JOB_ID}"
mkdir -p "$RELAYSPEC_OUTPUT"
cp SOURCE_COMMIT "$RELAYSPEC_OUTPUT/source-commit.txt"
sha256sum scripts/run_controlled_scaling.py scripts/train_relay.py scripts/benchmark_relay.py src/relayspec/*.py > "$RELAYSPEC_OUTPUT/source-sha256.txt"
tar -czf "$RELAYSPEC_OUTPUT/source-snapshot.tar.gz" src scripts configs slurm SOURCE_COMMIT
"$RELAYSPEC_PYTHON" scripts/check_allocation.py > "$RELAYSPEC_OUTPUT/allocation.jsonl"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used --format=csv -l 10 > "$RELAYSPEC_OUTPUT/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
pilot_args=()
if [[ -n "${CACHE_TRAINING_CONFIG:-}" ]]; then
  pilot_args+=(--training-config "$CACHE_TRAINING_CONFIG")
fi
if [[ -n "${CACHE_CAMPAIGN_CONFIG:-}" ]]; then
  pilot_args+=(--campaign-config "$CACHE_CAMPAIGN_CONFIG")
fi
timeout --signal=TERM --kill-after=10s 540s "$RELAYSPEC_PYTHON" scripts/run_cached_fit_pilot.py "${pilot_args[@]}"
