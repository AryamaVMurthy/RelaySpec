#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --job-name=matrix-batch-final
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
out="$root/batch-final-q8"
snapshot="$out/source-$SLURM_JOB_ID"
mkdir -p "$snapshot/experiments/handoff_transfer"
rsync -a --exclude __pycache__ "$control/experiments/handoff_transfer/matrix" "$snapshot/experiments/handoff_transfer/"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm" "$snapshot/experiments/"
export PYTHONPATH="$snapshot" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$out/gpu-$SLURM_JOB_ID.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python -m experiments.handoff_transfer.matrix.batch_final --root "$root" --models /scratch/aryama.murthy/relayspec-auf-20260911/models --target /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target --out "$out"
