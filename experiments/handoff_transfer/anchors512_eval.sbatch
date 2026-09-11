#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=handoff-anchors512-eval
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${FAMILY:?Set q8 or llama}"
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
run="$root/anchors512/$FAMILY"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
mkdir -p "$run/eval-source-$SLURM_JOB_ID/experiments" "$run/logs"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm/" "$run/eval-source-$SLURM_JOB_ID/experiments/auf_vllm/"
export PYTHONPATH="$run/eval-source-$SLURM_JOB_ID"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
"$study_python" -c "import json;assert json.load(open('$run/modules/full-fusion_r56/fusion_r56/verification.json'))['status']=='passed';assert json.load(open('$run/model-contract-fusion_r56.json'))['num_anchors']==512"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$run/logs/eval-gpu.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
extra=()
if [ "$FAMILY" = q8 ]; then extra+=(--target-path /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target); fi
for repeat in 0 1 2; do
  "$study_python" -u -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --models "$study/models" "${extra[@]}" --export "$run/exports/full-fusion_r56/fusion_r56" --data "$root/$FAMILY/evaluation" --mode handoff-r56 --count 128 --cap 2048 --repeat "$repeat" --out "$run/measurements"
  "$study_python" -m experiments.auf_vllm.compare_outputs --ar "$root/$FAMILY/measurements/ar-r$repeat-w0.jsonl" --method "$run/measurements/handoff-r56-r$repeat-w0.jsonl" --out "$run/comparisons/ar-r$repeat.json" --require-exact --expected-count 128
done
