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
#SBATCH --array=0-1%2
#SBATCH --job-name=auf-q8-dev128
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
# Keep each worker's preparation paths separate; both process the same128
# manifest partitioned by index. These are development measurements.
for objective in zip ce auf; do
  rate=$("$study_python" -m experiments.auf_vllm.select_lr --root "$study_root/lr-screen" --out "$study_root/lr-selection-decode-${SLURM_ARRAY_TASK_ID}.json" --loss "$objective")
  case="$study_root/benchmarks/q8-selected/$objective/worker-${SLURM_ARRAY_TASK_ID}"
  mkdir -p "$case"
  if [ ! -e "$case/export" ]; then ln -s "$study_root/fits/q8-n4096-${objective}-lr${rate}-s42/epoch-3/export" "$case/export"; fi
  TRANSFER_WORK="$study_original" "$study_python" -m experiments.auf_vllm.prepare_benchmark --case "$case"
  TRANSFER_WORK="$case" "$study_python" -u experiments/auf_vllm/runtime_zip/benchmark.py mapped --count 128 --cap 2048 --workers 2 --worker-index "$SLURM_ARRAY_TASK_ID" --tag development128-cap2048
  if [ "$objective" = zip ]; then
    for mode in native8 ar8; do
      TRANSFER_WORK="$case" "$study_python" -u experiments/auf_vllm/runtime_zip/benchmark.py "$mode" --count 128 --cap 2048 --workers 2 --worker-index "$SLURM_ARRAY_TASK_ID" --tag development128-cap2048
    done
  fi
done
