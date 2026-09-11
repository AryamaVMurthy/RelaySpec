#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=auf-fit-seeds
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
: "${FAMILY:?Set q8 or llama}"
study_snapshot="outputs/source-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$study_snapshot/experiments"
cp -r experiments/auf_vllm "$study_snapshot/experiments/"
export PYTHONPATH="$PWD/$study_snapshot/experiments/auf_vllm/runtime_zip:$PWD/$study_snapshot"
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
study_root=/scratch/aryama.murthy/relayspec-auf-20260911
losses=(zip ce auf)
loss=${losses[$((SLURM_ARRAY_TASK_ID / 2))]}
seed=$((43+SLURM_ARRAY_TASK_ID % 2))
fit="$study_root/fits/${FAMILY}-n4096-${loss}-lr1e-3-s${seed}"
cd "$study_snapshot"
if [ "$FAMILY" = q8 ]; then
  data="$study_root/main-q8-n4096"
  valid="$study_root/validation-q8-n1024"
  if [ "$loss" = zip ]; then
    "$study_python" -u -m experiments.auf_vllm.train_zip --records 4096 --epochs 3 --seed "$seed" --lr 1e-3 --manifest "$data/train.json" --out "$fit"
  else
    "$study_python" -u -m experiments.auf_vllm.train_tokens --objective "$loss" --records 4096 --epochs 3 --seed "$seed" --lr 1e-3 --data "$data" --validation-data "$valid" --defer-validation --out "$fit"
  fi
  "$study_python" -u -m experiments.auf_vllm.evaluate_offline --fit "$fit" --data "$valid"
  target_args=(--target-path "$TRANSFER_WORK/models/8b/target")
else
  test "$FAMILY" = llama
  data="$study_root/main-l3-n4096"
  valid="$study_root/validation-llama-n1024"
  "$study_python" -u -m experiments.auf_vllm.family_training --family llama --models "$study_root/models" --data "$data" --validation-manifest "$valid/combined-${loss}.json" --records 4096 --epochs 3 --seed "$seed" --objective "$loss" --lr 1e-3 --out "$fit"
  "$study_python" -u -m experiments.auf_vllm.family_offline --family llama --models "$study_root/models" --data "$valid" --fit "$fit"
  target_args=()
fi
export VLLM_BATCH_INVARIANT=1 OMP_NUM_THREADS=1
data="$study_root/benchmarks/workloads/${FAMILY}/math/development"
out="$study_root/benchmarks/seed-results/${FAMILY}/seed-${seed}"
"$study_python" -u -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --models "$study_root/models" "${target_args[@]}" --data "$data" --mode "$loss" --export "$fit/epoch-3/export" --count 128 --cap 2048 --out "$out"
"$study_python" -m experiments.auf_vllm.compare_outputs --ar "$study_root/benchmarks/workload-results/${FAMILY}/math/ar-r0-w0.jsonl" --method "$out/${loss}-r0-w0.jsonl" --out "$out/${loss}-comparison.json" --expected-count 128
