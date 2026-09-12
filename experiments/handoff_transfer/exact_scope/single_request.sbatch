#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
worker=$((SLURM_ARRAY_TASK_ID % 4))
family=q8
if ((SLURM_ARRAY_TASK_ID >= 4)); then family=cross; fi
out="$root/single128e1-20260912/$family/worker-$worker"
snapshot="$out/source-$SLURM_JOB_ID"
mkdir -p "$snapshot/experiments" "$snapshot/src"
rsync -a --exclude reports --exclude __pycache__ --exclude overlay --exclude specforge --exclude vendor --exclude data "$control/experiments/auf_vllm" "$control/experiments/handoff_transfer" "$snapshot/experiments/"
rsync -a --exclude __pycache__ "$control/src/relayspec" "$snapshot/src/"
export PYTHONPATH="$snapshot/src:$snapshot" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
cd "$snapshot"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$out/gpu-$SLURM_JOB_ID.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
"$python" -u -m experiments.handoff_transfer.exact_scope.single_request --family "$family" --worker "$worker"
mkdir -p "$control/outputs/exact32e1b8/single-request/$family/worker-$worker"
cp "$out/complete.json" "$out/progress.json" "$out/gpu-$SLURM_JOB_ID.csv" "$control/outputs/exact32e1b8/single-request/$family/worker-$worker/"
