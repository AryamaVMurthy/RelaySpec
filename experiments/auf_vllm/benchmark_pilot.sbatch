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
#SBATCH --job-name=auf-pilot-decode
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
: "${FIT_JOB:?Set completed pilot fit job ID}"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false VLLM_BATCH_INVARIANT=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/experiments/auf_vllm/runtime_zip:$PWD"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
study_original=/scratch/aryama.murthy/transfer-reproduction-20260907/work
study_fit=/scratch/aryama.murthy/relayspec-auf-20260911/pilot-fit-${FIT_JOB}
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 1 > "logs/gpu-${SLURM_JOB_ID}.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
for objective in auf ce zip; do
  TRANSFER_WORK="$study_original" "$study_python" -m experiments.auf_vllm.prepare_benchmark --case "$study_fit/$objective"
  TRANSFER_WORK="$study_fit/$objective" "$study_python" -u experiments/auf_vllm/runtime_zip/benchmark.py mapped --count 4 --cap 128 --tag diagnostic4-cap128
done
for mode in ar8 native8; do
  TRANSFER_WORK="$study_fit/auf" "$study_python" -u experiments/auf_vllm/runtime_zip/benchmark.py "$mode" --count 4 --cap 128 --tag diagnostic4-cap128
done
