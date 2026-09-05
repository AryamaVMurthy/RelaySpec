#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --nodelist=node07
#SBATCH --job-name=rs-sd2-setup
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${RELAYSPEC_PROJECT_DIR:?immutable source required}"
export RELAYSPEC_CACHE_DIR="/scratch/$USER/factorspec-runtime-20260826"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export RELAYSPEC_OUTPUT="$PWD/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
export SD_SQUARE_OVERLAY="$RELAYSPEC_CACHE_DIR/sd-square-python-overlay-$SLURM_JOB_ID"
export TMP_PATH="$RELAYSPEC_CACHE_DIR/sd-square-setup-$SLURM_JOB_ID"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 WANDB_MODE=disabled
mkdir -p "$RELAYSPEC_OUTPUT" "$TMP_PATH"
cp SOURCE_COMMIT "$RELAYSPEC_OUTPUT/source-commit.txt"
scontrol show job "$SLURM_JOB_ID" > "$RELAYSPEC_OUTPUT/allocation.txt"
timeout --signal=TERM --kill-after=10s 540s "$HOME/.venvs/factorspec-20260826/bin/python" \
  scripts/setup_sd_square.py --install --config configs/submission/baselines/sd-square-compatibility.json
