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
#SBATCH --job-name=rs-continuation
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module load u22/cuda/12.4
cd "${RELAYSPEC_PROJECT_DIR:?explicit immutable snapshot required}"
export RELAYSPEC_PYTHON="$HOME/.venvs/factorspec-20260826/bin/python"
export RELAYSPEC_CACHE_DIR="/scratch/$USER/factorspec-runtime-20260826"
export PYTHONPATH="$PWD/src:$HOME/factorspec/vendor/dflash"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export RELAYSPEC_OUTPUT="$PWD/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
export RELAYSPEC_FIT_OUTPUT="$RELAYSPEC_CACHE_DIR/relayspec/cached-fits/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
: "${RELAYSPEC_FEATURE_CACHE:?explicit verified cache required}"
: "${RELAYSPEC_PILOT_GATE:?explicit capacity gate required}"
: "${RELAYSPEC_CACHE_ACCESS_GATE:?explicit cache access gate required}"
mkdir -p "$RELAYSPEC_OUTPUT"
cp SOURCE_COMMIT "$RELAYSPEC_OUTPUT/source-commit.txt"
tar -czf "$RELAYSPEC_OUTPUT/source-snapshot.tar.gz" src scripts configs slurm SOURCE_COMMIT
"$RELAYSPEC_PYTHON" scripts/check_allocation.py > "$RELAYSPEC_OUTPUT/allocation.jsonl"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used --format=csv -l 10 > "$RELAYSPEC_OUTPUT/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
timeout --signal=TERM --kill-after=10s 540s "$RELAYSPEC_PYTHON" scripts/run_continuation_pilot.py
