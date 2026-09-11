#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node07
#SBATCH --time=03:00:00
#SBATCH --job-name=handoff-draft-eval
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${AUF_DRAFT_RANK:?Set rank}"
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
run="$root/q8-draft-control-r$AUF_DRAFT_RANK"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
mkdir -p "$run/eval-source-$SLURM_JOB_ID/experiments" "$run/logs"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm/" "$run/eval-source-$SLURM_JOB_ID/experiments/auf_vllm/"
export PYTHONPATH="$run/eval-source-$SLURM_JOB_ID"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
"$study_python" -c "import json;assert json.load(open('$run/modules/full-draft_lora/draft_lora/verification.json'))['status']=='passed';assert json.load(open('$run/model-contract-draft_lora.json'))['rank']==int('$AUF_DRAFT_RANK')"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$run/logs/eval-gpu.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
for mode in ar handoff-draft; do
  extra=()
  if [ "$mode" = handoff-draft ]; then extra+=(--export "$run/exports/full-draft_lora/draft_lora"); fi
  "$study_python" -u -m experiments.auf_vllm.family_benchmark --family q8 --models "$study/models" --target-path /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target "${extra[@]}" --data "$root/q8/evaluation" --mode "$mode" --count 4 --cap 128 --repeat 0 --out "$run/gate-$SLURM_JOB_ID"
done
"$study_python" -m experiments.auf_vllm.compare_outputs --ar "$run/gate-$SLURM_JOB_ID/ar-r0-w0.jsonl" --method "$run/gate-$SLURM_JOB_ID/handoff-draft-r0-w0.jsonl" --out "$run/gate-$SLURM_JOB_ID/comparison.json" --require-exact --expected-count 4
for repeat in 0 1 2; do
  "$study_python" -u -m experiments.auf_vllm.family_benchmark --family q8 --models "$study/models" --target-path /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target --export "$run/exports/full-draft_lora/draft_lora" --data "$root/q8/evaluation" --mode handoff-draft --count 128 --cap 2048 --repeat "$repeat" --out "$run/measurements"
  "$study_python" -m experiments.auf_vllm.compare_outputs --ar "$root/q8/measurements/ar-r$repeat-w0.jsonl" --method "$run/measurements/handoff-draft-r$repeat-w0.jsonl" --out "$run/comparisons/ar-r$repeat.json" --require-exact --expected-count 128
done
