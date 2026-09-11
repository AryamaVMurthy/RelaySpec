#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=matrix-batch-gate
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
run="$root/matrix/q8/five_ba56-auf-lr0.0001"
out="$root/batch-gate-$SLURM_JOB_ID"
mkdir -p "$out/source/experiments"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm" "$out/source/experiments/"
export PYTHONPATH="$out/source" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
"$study_python" -c "import json;assert json.load(open('$run/modules/steps-100/five_ba56/verification.json'))['status']=='passed'"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$out/gpu.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
for batch in 4 8; do
 for mode in ar matrix; do
  extra=()
  if [ "$mode" = matrix ]; then extra+=(--export "$run/exports/steps-100/five_ba56"); fi
  "$study_python" -m experiments.auf_vllm.family_benchmark --family q8 --target-path /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target --models "$study/models" --data "$run/validation-data-31784" --mode "$mode" --count 8 --cap 128 --request-batch-size "$batch" --out "$out" "${extra[@]}"
 done
 "$study_python" -m experiments.auf_vllm.compare_outputs --ar "$out/ar-r0-w0-b$batch.jsonl" --method "$out/matrix-r0-w0-b$batch.jsonl" --out "$out/comparison-b$batch.json" --expected-count 8 --require-exact
done
