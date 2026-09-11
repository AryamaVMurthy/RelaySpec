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
#SBATCH --job-name=auf-direct-fusion
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
: "${STAGE:?Set screen or main}"
study_snapshot="outputs/source-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$study_snapshot/experiments"
cp -r experiments/auf_vllm "$study_snapshot/experiments/"
export PYTHONPATH="$PWD/$study_snapshot/experiments/auf_vllm/runtime_zip:$PWD/$study_snapshot"
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
study_root=/scratch/aryama.murthy/relayspec-auf-20260911
losses=(ce auf)
rates=(1e-4 3e-4 1e-3)
cd "$study_snapshot"
if [ "$STAGE" = screen ]; then
  loss=${losses[$((SLURM_ARRAY_TASK_ID / 3))]}
  rate=${rates[$((SLURM_ARRAY_TASK_ID % 3))]}
  records=512
else
  test "$STAGE" = main
  loss=${losses[$((SLURM_ARRAY_TASK_ID % 2))]}
  counts=(512 4096)
  records=${counts[$((SLURM_ARRAY_TASK_ID / 2))]}
  rate=$("$study_python" -m experiments.auf_vllm.select_draft_lora_lr --root "$study_root/direct-screen" --location direct --loss "$loss" --rates 1e-4 3e-4 1e-3 --out "$study_root/direct-selection-${loss}.json")
fi
fit="$study_root/direct-screen/direct-${loss}-lr${rate}"
if [ "$records" = 4096 ]; then fit="$study_root/direct-main/direct-${loss}-lr${rate}";fi
# The selected512 screen fit is reused by the main evaluation; no extra epochs.
"$study_python" -u -m experiments.auf_vllm.train_tokens --objective "$loss" --records "$records" --epochs 3 --seed 42 --lr "$rate" --direct-fusion --data "$study_root/main-q8-n4096" --validation-data "$study_root/validation-q8-n1024" --defer-validation --out "$fit"
if [ "$STAGE" = screen ] || [ "$records" = 4096 ]; then
  "$study_python" -u -m experiments.auf_vllm.evaluate_offline --fit "$fit" --data "$study_root/validation-q8-n1024"
fi
if [ "$STAGE" = main ]; then
  case="$study_root/benchmarks/direct-fusion/n${records}-${loss}"
  mkdir -p "$case"
  if [ ! -e "$case/export" ]; then ln -s "$fit/epoch-3/export" "$case/export";fi
  "$study_python" -m experiments.auf_vllm.prepare_benchmark --case "$case"
  export VLLM_BATCH_INVARIANT=1 OMP_NUM_THREADS=1
  TRANSFER_WORK="$case" "$study_python" -u experiments/auf_vllm/runtime_zip/benchmark.py mapped --count 128 --cap 2048 --tag development128-cap2048
  "$study_python" -m experiments.auf_vllm.compare_outputs --ar "$study_root/benchmarks/q8-selected/zip/worker-0/measurements/development128-cap2048/worker_0/ar8-r0.jsonl" "$study_root/benchmarks/q8-selected/zip/worker-1/measurements/development128-cap2048/worker_1/ar8-r0.jsonl" --method "$case/measurements/development128-cap2048/worker_0/mapped-r0.jsonl" --out "$case/comparison.json" --expected-count 128
fi
