#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node07
#SBATCH --time=01:30:00
#SBATCH --array=0-2%2
#SBATCH --job-name=auf-q8-selectedfit
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
losses=(zip ce auf)
LOSS=${losses[$SLURM_ARRAY_TASK_ID]}
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
# Snapshot the code at job start; later development cannot change this run.
study_snapshot="outputs/source-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$study_snapshot/experiments"
cp -r experiments/auf_vllm "$study_snapshot/experiments/"
export PYTHONPATH="$PWD/$study_snapshot"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
study_data=/scratch/aryama.murthy/relayspec-auf-20260911/main-q8-n4096
study_valid=/scratch/aryama.murthy/relayspec-auf-20260911/validation-q8-n1024
study_root=/scratch/aryama.murthy/relayspec-auf-20260911
LR=$("$study_python" -m experiments.auf_vllm.select_lr --root "$study_root/lr-screen" --out "$study_root/lr-selection-${LOSS}.json" --loss "$LOSS")
study_out="$study_root/fits/q8-n4096-${LOSS}-lr${LR}-s42"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 1 > "logs/gpu-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
cd "$study_snapshot"
if [ "$LOSS" = zip ]; then
  "$study_python" -u -m experiments.auf_vllm.train_zip --records 4096 --epochs 3 --seed 42 --lr "$LR" --manifest "$study_data/train.json" --out "$study_out"
else
  "$study_python" -u -m experiments.auf_vllm.train_tokens --objective "$LOSS" --records 4096 --epochs 3 --seed 42 --lr "$LR" --microbatch-records 4 --logical-records 32 --data "$study_data" --validation-data "$study_valid" --defer-validation --out "$study_out"
fi
if [ ! -f "$study_out/offline-validation.json" ]; then
  "$study_python" -u -m experiments.auf_vllm.evaluate_offline --fit "$study_out" --data "$study_valid"
fi
