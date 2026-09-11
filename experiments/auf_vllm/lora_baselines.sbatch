#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node07
#SBATCH --time=04:00:00
#SBATCH --job-name=auf-lora-controls
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
study_snapshot="outputs/source-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$study_snapshot/experiments"
cp -r experiments/auf_vllm "$study_snapshot/experiments/"
export PYTHONPATH="$PWD/$study_snapshot/experiments/auf_vllm/runtime_zip:$PWD/$study_snapshot"
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
study_root=/scratch/aryama.murthy/relayspec-auf-20260911
names=(zip-base native-base)
modes=(mapped native8)
case="$study_root/benchmarks/draft-lora/${names[$SLURM_ARRAY_TASK_ID]}"
mode=${modes[$SLURM_ARRAY_TASK_ID]}
mkdir -p "$case"
if [ ! -e "$case/export" ]; then
  ln -s "$study_root/fits/q8-n4096-zip-lr1e-3-s42/epoch-3/export" "$case/export"
fi
cd "$study_snapshot"
"$study_python" -m experiments.auf_vllm.prepare_benchmark --case "$case"
for repeat in 0 1 2; do
  TRANSFER_WORK="$case" "$study_python" -u experiments/auf_vllm/runtime_zip/benchmark.py "$mode" --count 128 --cap 2048 --workers 1 --worker-index 0 --repeat "$repeat" --tag development128-cap2048
done
