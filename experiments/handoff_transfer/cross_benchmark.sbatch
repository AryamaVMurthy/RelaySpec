#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --job-name=cross-benchmark
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${EVAL_SCOPE:?screen or final}" "${MODE:?ar, zip, normal, or matrix}"
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
snapshot="$root/cross-evaluation-sources/$SLURM_JOB_ID"
mkdir -p "$snapshot/experiments/handoff_transfer" "$snapshot/logs"
rsync -a --exclude __pycache__ "$control/experiments/handoff_transfer/cross" "$control/experiments/handoff_transfer/matrix" "$snapshot/experiments/handoff_transfer/"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm" "$snapshot/experiments/"
export PYTHONPATH="$snapshot" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$snapshot/logs/gpu.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
export CROSS_SOURCE_TOKENIZER="$study/models/qwen4-source"
target="$study/models/llama8-source"
base="$root/cross"
extra=()
case "$MODE" in
 matrix)
  : "${KIND:?}" "${MATRIX_OBJECTIVE:?}" "${MATRIX_LR:?}" "${TRAIN_STEPS:?}"
  run="$root/matrix/cross/$KIND-$MATRIX_OBJECTIVE-lr$MATRIX_LR"
  "$study_python" -c "import json;v=json.load(open('$run/modules/steps-$TRAIN_STEPS/$KIND/verification.json'));assert v['status']=='passed'"
  extra+=(--export "$run/exports/steps-$TRAIN_STEPS/$KIND");;
 ar) ;;
 zip) extra+=(--export "$root/cross-full4096/initializers/epoch-3/export");;
 normal) extra+=(--export "$root/cross-full4096/initializers/normal/export");;
 *) exit 2;;
esac
if [ "$EVAL_SCOPE" = screen ]; then
 test "$MODE" = matrix
 data="$run/validation-data-$SLURM_JOB_ID";out="$run/screen-$TRAIN_STEPS"
 "$study_python" -m experiments.handoff_transfer.matrix.validation --family cross --target "$target" --source "$study/manifests/eval.jsonl" --train "$study/manifests/train.jsonl" --final "$study/manifests/dev.jsonl" --out "$data"
 "$study_python" -m experiments.auf_vllm.family_benchmark --family cross --target-path "$target" --models "$study/models" --data "$data" --mode ar --count 32 --cap 512 --out "$out"
 "$study_python" -m experiments.auf_vllm.family_benchmark --family cross --target-path "$target" --models "$study/models" --data "$data" "${extra[@]}" --mode matrix --count 32 --cap 512 --out "$out"
 "$study_python" -m experiments.auf_vllm.compare_outputs --ar "$out/ar-r0-w0.jsonl" --method "$out/matrix-r0-w0.jsonl" --out "$out/comparison.json" --expected-count 32 --require-exact
elif [ "$EVAL_SCOPE" = final ]; then
 data="$base/evaluation"
 "$study_python" -m experiments.auf_vllm.prepare_family_benchmark --family cross --target-path "$target" --models "$study/models" --manifest "$study/manifests/dev.jsonl" --out "$data"
 out="$base/measurements"
 if [ "$MODE" = matrix ]; then test "$TRAIN_STEPS" = 2000;out="$run/measurements";fi
 for repeat in 0 1 2; do
  "$study_python" -m experiments.auf_vllm.family_benchmark --family cross --target-path "$target" --models "$study/models" --data "$data" "${extra[@]}" --mode "$MODE" --count 128 --cap 2048 --repeat "$repeat" --out "$out"
 done
 if [ "$MODE" = matrix ]; then
  "$study_python" -m experiments.handoff_transfer.matrix.collect --run "$run" --baseline "$base" --out "$run/final-comparison.json"
 fi
else
 exit 2
fi
