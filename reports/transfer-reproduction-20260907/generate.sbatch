#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --nodelist=node07
#SBATCH --time=03:00:00
#SBATCH --job-name=rs-transfer-generate
#SBATCH --output=/home/aryama.murthy/relayspec-transfer-reproduction/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-transfer-reproduction/logs/%x-%j.err
set -euo pipefail
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
cd /scratch/aryama.murthy/transfer-reproduction-20260907/package
printf 'start=%s job=%s gpu=%s\n' "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
.venv/bin/python verify_bundle.py
.venv/bin/python reproduce.py approve-generation
nvidia-smi --query-gpu=timestamp,index,name,driver_version,utilization.gpu,memory.used --format=csv -l 30 > /home/aryama.murthy/relayspec-transfer-reproduction/logs/gpu-${SLURM_JOB_ID}.csv &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
.venv/bin/python reproduce.py generate --gpus "$CUDA_VISIBLE_DEVICES"
printf 'end=%s\n' "$(date --iso-8601=seconds)"
