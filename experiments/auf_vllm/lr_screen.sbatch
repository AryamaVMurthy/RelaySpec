#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node07
#SBATCH --array=0-8%2
#SBATCH --time=00:15:00
#SBATCH --job-name=auf-lr-screen512
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
losses=(zip ce auf)
rates=(1e-4 3e-4 1e-3)
loss=${losses[$((SLURM_ARRAY_TASK_ID / 3))]}
rate=${rates[$((SLURM_ARRAY_TASK_ID % 3))]}
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
study_snapshot="outputs/source-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$study_snapshot/experiments"
cp -r experiments/auf_vllm "$study_snapshot/experiments/"
export PYTHONPATH="$PWD/$study_snapshot"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
study_root=/scratch/aryama.murthy/relayspec-auf-20260911
study_out="$study_root/lr-screen/q8-n512-${loss}-lr${rate}-s42"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 1 > "logs/gpu-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
cd "$study_snapshot"
if [ "$loss" = zip ]; then
  "$study_python" -u -m experiments.auf_vllm.train_zip --records 512 --epochs 3 --lr "$rate" --manifest "$study_root/main-q8-n4096/train.json" --out "$study_out"
else
  "$study_python" -u -m experiments.auf_vllm.train_tokens --objective "$loss" --records 512 --epochs 3 --lr "$rate" --data "$study_root/main-q8-n4096" --validation-data "$study_root/validation-q8-n1024" --defer-validation --out "$study_out"
fi
"$study_python" -u -m experiments.auf_vllm.evaluate_offline --fit "$study_out" --data "$study_root/validation-q8-n1024"
