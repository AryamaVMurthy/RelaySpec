#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=handoff-compute-eval
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
: "${FAMILY:?Set q8 or q14}"
control=/home/aryama.murthy/relayspec-auf-20260911
code="$control/experiments/handoff_transfer"
root=/scratch/aryama.murthy/handoff-transfer-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export AUF_WORKDIR="$root/$FAMILY" AUF_MODELS="$root/models/qwen4" AUF_DOMAIN=math
export SPECFORGE_ROOT="$code/specforge"
export PYTHONPATH="$code/port:$code/vendor/code:$code/overlay:$SPECFORGE_ROOT:$control"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
kinds=(fusion_r56 five_maps)
modes=(handoff-r56 handoff-five)
kind=${kinds[$((SLURM_ARRAY_TASK_ID / 3))]}
mode=${modes[$((SLURM_ARRAY_TASK_ID / 3))]}
steps=$(( (SLURM_ARRAY_TASK_ID % 3 + 1) * 500 ))
seen=$(( steps * 8 ))
model="$AUF_WORKDIR/exports/full-$kind/seen_$seen"
module="$AUF_WORKDIR/modules/full-$kind/seen_$seen"
"$study_python" "$code/verify_transfer.py" --kind "$kind" --tag "seen_$seen" --steps "$steps" --module-dir "$module" --export-dir "$model"
out="$AUF_WORKDIR/compute/updates-$steps"
mkdir -p "$out/logs"
extra=()
if [ "$FAMILY" = q8 ]; then extra+=(--target-path /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target); fi
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$out/logs/$kind.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
for repeat in 0 1 2; do
  "$study_python" -u -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --models "$study/models" "${extra[@]}" --export "$model" --data "$AUF_WORKDIR/evaluation" --mode "$mode" --count 128 --cap 2048 --repeat "$repeat" --out "$out/measurements"
  "$study_python" -m experiments.auf_vllm.compare_outputs --ar "$AUF_WORKDIR/measurements/ar-r$repeat-w0.jsonl" --method "$out/measurements/$mode-r$repeat-w0.jsonl" --out "$out/comparisons/$mode-r$repeat.json" --require-exact --expected-count 128
done
