#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node07
#SBATCH --time=08:00:00
#SBATCH --job-name=auf-q8-fit32k
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
study_snapshot="outputs/source-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$study_snapshot/experiments"
cp -r experiments/auf_vllm "$study_snapshot/experiments/"
export PYTHONPATH="$PWD/$study_snapshot/experiments/auf_vllm/runtime_zip:$PWD/$study_snapshot"
study_root=/scratch/aryama.murthy/relayspec-auf-20260911
export TRANSFER_WORK="$study_root/main-q8-n32768"
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
losses=(zip ce auf)
loss=${losses[$SLURM_ARRAY_TASK_ID]}
fit="$study_root/fits/q8-n32768-${loss}-lr1e-3-s42"
valid="$study_root/validation-q8-n1024"
cd "$study_snapshot"
if [ "$loss" = zip ]; then
  "$study_python" -u -m experiments.auf_vllm.train_zip --records 32768 --epochs 3 --seed 42 --lr 1e-3 --manifest "$TRANSFER_WORK/train.json" --out "$fit"
else
  "$study_python" -u -m experiments.auf_vllm.train_tokens --objective "$loss" --records 32768 --epochs 3 --seed 42 --lr 1e-3 --data "$TRANSFER_WORK" --validation-data "$valid" --defer-validation --out "$fit"
fi
"$study_python" -u -m experiments.auf_vllm.evaluate_offline --fit "$fit" --data "$valid"
case="$study_root/benchmarks/q8-n32768/$loss"
mkdir -p "$case"
if [ ! -e "$case/export" ]; then ln -s "$fit/epoch-3/export" "$case/export";fi
"$study_python" -m experiments.auf_vllm.prepare_benchmark --case "$case"
export VLLM_BATCH_INVARIANT=1 OMP_NUM_THREADS=1
TRANSFER_WORK="$case" "$study_python" -u experiments/auf_vllm/runtime_zip/benchmark.py mapped --count 128 --cap 2048 --tag development128-cap2048
"$study_python" -m experiments.auf_vllm.compare_outputs --ar "$study_root/benchmarks/q8-selected/zip/worker-0/measurements/development128-cap2048/worker_0/ar8-r0.jsonl" "$study_root/benchmarks/q8-selected/zip/worker-1/measurements/development128-cap2048/worker_1/ar8-r0.jsonl" --method "$case/measurements/development128-cap2048/worker_0/mapped-r0.jsonl" --out "$case/comparison.json" --expected-count 128
