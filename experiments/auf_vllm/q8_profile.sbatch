#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node07
#SBATCH --time=00:15:00
#SBATCH --array=0-4%2
#SBATCH --job-name=auf-q8-gpu-profile
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911

export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false VLLM_BATCH_INVARIANT=1 HF_HUB_OFFLINE=1
study_snapshot="outputs/source-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$study_snapshot/experiments"
cp -r experiments/auf_vllm "$study_snapshot/experiments/"
export PYTHONPATH="$PWD/$study_snapshot/experiments/auf_vllm/runtime_zip:$PWD/$study_snapshot"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
study_original=/scratch/aryama.murthy/transfer-reproduction-20260907/work
study_root=/scratch/aryama.murthy/relayspec-auf-20260911
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 1 > "logs/gpu-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
cd "$study_snapshot"
modes=(ar8 native8 zip ce auf)
mode=${modes[$SLURM_ARRAY_TASK_ID]}
loss="$mode"
if [ "$mode" = ar8 ] || [ "$mode" = native8 ]; then loss=zip; else mode=mapped; fi
case="$study_root/benchmarks/q8-selected/$loss/worker-0"
profile="$study_root/profiles/q8/${modes[$SLURM_ARRAY_TASK_ID]}-${SLURM_ARRAY_JOB_ID}"
TRANSFER_WORK="$case" "$study_python" -u experiments/auf_vllm/runtime_zip/benchmark.py "$mode" --count 4 --cap 128 --tag "profile-${SLURM_ARRAY_JOB_ID}" --profile-dir "$profile"
