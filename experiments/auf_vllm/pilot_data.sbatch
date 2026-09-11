#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node07
#SBATCH --time=00:20:00
#SBATCH --job-name=auf-pilot-data
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false VLLM_BATCH_INVARIANT=1
export PYTHONPATH="$PWD/experiments/auf_vllm/bootstrap:$PWD"
export HF_HUB_OFFLINE=1
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
study_data=/scratch/aryama.murthy/relayspec-auf-20260911/pilot-q8
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 1 > "logs/gpu-${SLURM_JOB_ID}.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
"$study_python" -u -m experiments.auf_vllm.pilot_data prepare --out "$study_data"
"$study_python" -u -m experiments.auf_vllm.pilot_data dev --out "$study_data"
AUF_CAPTURE=1 "$study_python" -u -m experiments.auf_vllm.pilot_data capture --size 8 --out "$study_data"
AUF_CAPTURE=1 "$study_python" -u -m experiments.auf_vllm.pilot_data capture --size 4 --out "$study_data"
