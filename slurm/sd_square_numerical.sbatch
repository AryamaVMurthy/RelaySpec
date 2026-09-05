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
#SBATCH --job-name=rs-sd2-numerical
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module load u22/cuda/12.4
cd "${RELAYSPEC_PROJECT_DIR:?immutable source required}"
: "${SD_SQUARE_SETUP_GATE:?verified CPU setup required}"
: "${SD_SQUARE_PRIOR_OUTPUT:?original failed pilot required}"
export RELAYSPEC_PYTHON="$HOME/.venvs/factorspec-20260826/bin/python"
export SD_SQUARE_OVERLAY
SD_SQUARE_OVERLAY="$("$RELAYSPEC_PYTHON" -c 'import json,sys; g=json.load(open(sys.argv[1])); assert g["status"]=="pass"; print(g["overlay"])' "$SD_SQUARE_SETUP_GATE")"
export RELAYSPEC_CACHE_DIR="/scratch/$USER/factorspec-runtime-20260826"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$SD_SQUARE_OVERLAY:$PWD/src"
export TMP_PATH="$RELAYSPEC_CACHE_DIR/sd-square-pilot-$SLURM_JOB_ID"
export RELAYSPEC_FIT_OUTPUT="$RELAYSPEC_CACHE_DIR/relayspec/sd-square-fits/$SLURM_JOB_ID"
export RELAYSPEC_FEATURE_CACHE="$RELAYSPEC_CACHE_DIR/relayspec/feature-caches/numina-dflash8b-27723"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export RELAYSPEC_OUTPUT="$PWD/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RELAYSPEC_OUTPUT" "$TMP_PATH" "$RELAYSPEC_FIT_OUTPUT"
cp SOURCE_COMMIT "$RELAYSPEC_OUTPUT/source-commit.txt"
cp "$SD_SQUARE_SETUP_GATE" "$RELAYSPEC_OUTPUT/setup-gate.json"
tar -czf "$RELAYSPEC_OUTPUT/source-snapshot.tar.gz" src scripts configs slurm vendor/sd-square SOURCE_COMMIT
"$RELAYSPEC_PYTHON" scripts/check_allocation.py > "$RELAYSPEC_OUTPUT/allocation.jsonl"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used --format=csv -l 10 > "$RELAYSPEC_OUTPUT/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
timeout --signal=TERM --kill-after=10s 540s "$RELAYSPEC_PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=4 scripts/pilot_sd_square_numerical.py --config configs/submission/baselines/sd-square-numerical-replay.json
"$RELAYSPEC_PYTHON" scripts/summarize_sd_square_numerical.py
