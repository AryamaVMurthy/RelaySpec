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
#SBATCH --job-name=auf-draft-lora
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
locations=(draft draft fusion fusion)
losses=(ce auf ce auf)
rates=(2e-5 1e-4 1e-3)
if [ "$STAGE" = screen ]; then
  cell=$((SLURM_ARRAY_TASK_ID / 3))
  rate=${rates[$((SLURM_ARRAY_TASK_ID % 3))]}
  records=512
else
  test "$STAGE" = main
  cell=$SLURM_ARRAY_TASK_ID
  records=4096
fi
location=${locations[$cell]}
loss=${losses[$cell]}
cd "$study_snapshot"
if [ "$STAGE" = main ]; then
  rate=$("$study_python" -m experiments.auf_vllm.select_draft_lora_lr --root "$study_root/draft-lora-screen" --location "$location" --loss "$loss" --out "$study_root/draft-lora-selection-${location}-${loss}.json")
fi
study_out="$study_root/draft-lora-${STAGE}/${location}-${loss}-lr${rate}"
extra=()
if [ "$location" = draft ]; then extra+=(--draft-lora); fi
"$study_python" -u -m experiments.auf_vllm.train_tokens --objective "$loss" --records "$records" --epochs 3 --seed 42 --lr "$rate" --microbatch-records 4 --logical-records 32 --data "$study_root/main-q8-n4096" --validation-data "$study_root/validation-q8-n1024" --defer-validation --lora-base "$study_root/fits/q8-n4096-zip-lr1e-3-s42/epoch-3/export/model.safetensors" --rank 32 "${extra[@]}" --out "$study_out"
"$study_python" -u -m experiments.auf_vllm.evaluate_offline --fit "$study_out" --data "$study_root/validation-q8-n1024"
if [ "$STAGE" = main ]; then
  case="$study_root/benchmarks/draft-lora/${location}-${loss}"
  mkdir -p "$case"
  if [ ! -e "$case/export" ]; then ln -s "$study_out/epoch-3/export" "$case/export"; fi
  "$study_python" -m experiments.auf_vllm.prepare_benchmark --case "$case"
  export VLLM_BATCH_INVARIANT=1 OMP_NUM_THREADS=1
  for repeat in 0 1 2; do
    TRANSFER_WORK="$case" "$study_python" -u experiments/auf_vllm/runtime_zip/benchmark.py mapped --count 128 --cap 2048 --workers 1 --worker-index 0 --repeat "$repeat" --tag development128-cap2048
  done
fi
