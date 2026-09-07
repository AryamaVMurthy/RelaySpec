#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --nodelist=node07
#SBATCH --time=01:40:00
#SBATCH --job-name=rs-transfer-fullcapture
#SBATCH --output=/home/aryama.murthy/relayspec-transfer-reproduction/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-transfer-reproduction/logs/%x-%j.err
set -euo pipefail
printf 'start=%s job=%s host=%s GPUs=%s\n' "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" "$(hostname)" "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=timestamp,index,name,driver_version,utilization.gpu,memory.used --format=csv -l 30 > /home/aryama.murthy/relayspec-transfer-reproduction/logs/gpu-${SLURM_JOB_ID}.csv &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
cd /scratch/aryama.murthy/transfer-reproduction-20260907/package
.venv/bin/python /home/aryama.murthy/relayspec-transfer-reproduction/full_run.py capture
