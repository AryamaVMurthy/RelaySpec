#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=handoff-confirmation
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
: "${FAMILY:?Set family}"
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
snapshot="$root/confirmation-source-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
mkdir -p "$snapshot/experiments" "$root/confirmation-logs"
rsync -a --exclude reports --exclude overlay --exclude specforge --exclude __pycache__ "$control/experiments/handoff_transfer/" "$snapshot/experiments/handoff_transfer/"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm/" "$snapshot/experiments/auf_vllm/"
export PYTHONPATH="$snapshot"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
workloads=(math gsm code chat)
workload=${workloads[$((SLURM_ARRAY_TASK_ID / 3))]}
repeat=$((SLURM_ARRAY_TASK_ID % 3))
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$root/confirmation-logs/$FAMILY-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
"$study_python" -u -m experiments.handoff_transfer.confirmation_runner --protocol "$control/outputs/handoff-transfer/frozen-$FAMILY.json" --family "$FAMILY" --workload "$workload" --repeat "$repeat" --manifest "$study/workloads-20260911" --models "$study/models" --out "$root/confirmation"
