#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node07
#SBATCH --time=00:10:00
#SBATCH --job-name=auf-updates-gate
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
"$study_python" -u -m experiments.auf_vllm.fit_updates --objective auf --data "$study_root/main-q8-n4096" --validation-data "$study_root/validation-q8-n1024" --out "$study_root/updates-gate-${SLURM_JOB_ID}" --records 16 --updates 1024 --chunk-updates 16 --lr 1e-3
# Resume the same optimization trajectory, preserving schedule and data order.
"$study_python" -u -m experiments.auf_vllm.fit_updates --objective auf --data "$study_root/main-q8-n4096" --validation-data "$study_root/validation-q8-n1024" --out "$study_root/updates-gate-${SLURM_JOB_ID}" --records 16 --updates 1024 --chunk-updates 16 --lr 1e-3
