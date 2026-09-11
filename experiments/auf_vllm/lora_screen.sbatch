#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node07
#SBATCH --time=00:25:00
#SBATCH --job-name=auf-lora-screen512
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
study_snapshot="outputs/source-${SLURM_JOB_ID}"
mkdir -p "$study_snapshot/experiments"
cp -r experiments/auf_vllm "$study_snapshot/experiments/"
export PYTHONPATH="$PWD/$study_snapshot"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
study_root=/scratch/aryama.murthy/relayspec-auf-20260911

nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 1 > "logs/gpu-${SLURM_JOB_ID}.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
cd "$study_snapshot"
for loss in auf ce; do
  study_out="$study_root/lora-screen/q8-n512-${loss}-r56-lr1e-4-s42"
  "$study_python" -u -m experiments.auf_vllm.train_tokens --objective "$loss" --records 512 --epochs 3 --lr 1e-4 --data "$study_root/main-q8-n4096" --validation-data "$study_root/validation-q8-n1024" --defer-validation --lora-base "$study_root/fits/q8-n4096-zip-lr1e-3-s42/epoch-3/export/model.safetensors" --rank 56 --out "$study_out"
  "$study_python" -u -m experiments.auf_vllm.evaluate_offline --fit "$study_out" --data "$study_root/validation-q8-n1024"
done
