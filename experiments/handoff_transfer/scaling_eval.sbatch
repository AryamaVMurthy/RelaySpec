#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node07
#SBATCH --time=06:00:00
#SBATCH --job-name=handoff-scaling-eval
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
: "${FIT_JOB:?Set scaling fit job}" "${RECORDS:?Set records}" "${UPDATES:=1024}"
control=/home/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
run="$root/scaling/q8-n$RECORDS-u$UPDATES-$FIT_JOB"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
snapshot="$run/eval-source-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
mkdir -p "$snapshot/experiments" "$run/logs"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm/" "$snapshot/experiments/auf_vllm/"
export PYTHONPATH="$snapshot"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
modes=(zip handoff-r56 handoff-five)
kinds=(none fusion_r56 five_maps)
mode=${modes[$SLURM_ARRAY_TASK_ID]}
kind=${kinds[$SLURM_ARRAY_TASK_ID]}
if [ "$mode" = zip ]; then
  model="$root/scaling-initializers/q8-n$RECORDS-s42/epoch-3/export"
else
  "$study_python" -c "import json;assert json.load(open('$run/modules/full-$kind/$kind/verification.json'))['status']=='passed'"
  model="$run/exports/full-$kind/$kind"
fi
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$run/logs/eval-$mode.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
for repeat in 0 1 2; do
  "$study_python" -u -m experiments.auf_vllm.family_benchmark --family q8 --models "$study/models" --target-path /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target --export "$model" --data "$root/q8/evaluation" --mode "$mode" --count 128 --cap 2048 --repeat "$repeat" --out "$run/measurements"
  "$study_python" -m experiments.auf_vllm.compare_outputs --ar "$root/q8/measurements/ar-r$repeat-w0.jsonl" --method "$run/measurements/$mode-r$repeat-w0.jsonl" --out "$run/comparisons/$mode-r$repeat.json" --require-exact --expected-count 128
done
