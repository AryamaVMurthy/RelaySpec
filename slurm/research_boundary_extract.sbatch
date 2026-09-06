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
#SBATCH --job-name=rs-research-wave
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module load u22/cuda/12.4
: "${BOUNDARY_TRAINING_CONFIG:?Set BOUNDARY_TRAINING_CONFIG}"
: "${BOUNDARY_PILOT_GATE:?Set BOUNDARY_PILOT_GATE}"
cd "${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec-campaign-20260905}"
export RELAYSPEC_PYTHON="$HOME/.venvs/factorspec-20260826/bin/python"
export RELAYSPEC_CACHE_DIR="/scratch/$USER/factorspec-runtime-20260826"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export DFLASH_SOURCE="$HOME/factorspec/vendor/dflash"
export DEEPSPEC_SOURCE="$HOME/relayspec/vendor/deepspec"
export DEEPSPEC_PYTHON_OVERLAY="$RELAYSPEC_CACHE_DIR/deepspec-python-overlay-node07"
export PYTHONPATH="$PWD/src:$DFLASH_SOURCE"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export RELAYSPEC_OUTPUT="$PWD/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RELAYSPEC_OUTPUT"
cp SOURCE_COMMIT "$RELAYSPEC_OUTPUT/source-commit.txt"
sha256sum scripts/run_controlled_scaling.py scripts/train_relay.py scripts/benchmark_relay.py src/relayspec/*.py > "$RELAYSPEC_OUTPUT/source-sha256.txt"
tar -czf "$RELAYSPEC_OUTPUT/source-snapshot.tar.gz" src scripts configs slurm SOURCE_COMMIT
"$RELAYSPEC_PYTHON" scripts/check_allocation.py > "$RELAYSPEC_OUTPUT/allocation.jsonl"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used --format=csv -l 10 > "$RELAYSPEC_OUTPUT/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
export RELAYSPEC_FEATURE_CACHE="$RELAYSPEC_CACHE_DIR/relayspec/feature-caches/downscale-full-$SLURM_JOB_ID"
set +e
timeout --signal=TERM --kill-after=10s 540s "$RELAYSPEC_PYTHON" scripts/extract_scaling_cache.py --training-config "$BOUNDARY_TRAINING_CONFIG" --pilot-gate "$BOUNDARY_PILOT_GATE" --train-records 512 --validation-records 128 > "$RELAYSPEC_OUTPUT/pipeline.log" 2>&1
pipeline_code=$?
set -e
printf '%s\n' "$pipeline_code" > "$RELAYSPEC_OUTPUT/pipeline-exit-code.txt"
exit "$pipeline_code"
