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
#SBATCH --job-name=handoff-capacity-eval
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${AUF_CAPACITY_RANK:?Set rank}"
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
run="$root/q8-capacity-r$AUF_CAPACITY_RANK"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
mkdir -p "$run/eval-source-$SLURM_JOB_ID/experiments" "$run/logs"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm/" "$run/eval-source-$SLURM_JOB_ID/experiments/auf_vllm/"
export PYTHONPATH="$run/eval-source-$SLURM_JOB_ID"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
"$study_python" -c "import json;assert json.load(open('$run/modules/full-fusion_capacity/fusion_capacity/verification.json'))['status']=='passed';assert json.load(open('$run/model-contract-fusion_capacity.json'))['rank']==int('$AUF_CAPACITY_RANK')"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$run/logs/eval-gpu.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
for repeat in 0 1 2; do
  "$study_python" -u -m experiments.auf_vllm.family_benchmark --family q8 --models "$study/models" --target-path /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target --export "$run/exports/full-fusion_capacity/fusion_capacity" --data "$root/q8/evaluation" --mode handoff-capacity --count 128 --cap 2048 --repeat "$repeat" --out "$run/measurements"
  "$study_python" -m experiments.auf_vllm.compare_outputs --ar "$root/q8/measurements/ar-r$repeat-w0.jsonl" --method "$run/measurements/handoff-capacity-r$repeat-w0.jsonl" --out "$run/comparisons/ar-r$repeat.json" --require-exact --expected-count 128
done
