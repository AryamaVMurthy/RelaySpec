#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=handoff-cross-labels
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export PYTHONPATH="$control" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 VLLM_BATCH_INVARIANT=1
out=/scratch/aryama.murthy/handoff-transfer-20260911/anchors512/cross/pilot16
mkdir -p "$out"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw --format=csv -l 5 > "$out/gpu-$SLURM_JOB_ID.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid"' EXIT
"$study_python" -u -m experiments.handoff_transfer.cross.rollout --out "$out" --count 16
"$study_python" -u -m experiments.handoff_transfer.cross.prepare --out "$out" --models "$study/models"
