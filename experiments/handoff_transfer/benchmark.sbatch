#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --job-name=handoff-eval128
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${FAMILY:?Set family}" "${MODE:?Set ar, zip, handoff-r56 or handoff-five}"
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
run=/scratch/aryama.murthy/handoff-transfer-20260911/$FAMILY
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
snapshot="$run/source-eval-$SLURM_JOB_ID"
mkdir -p "$snapshot/experiments/auf_vllm" "$run/logs"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm/" "$snapshot/experiments/auf_vllm/"
export PYTHONPATH="$snapshot"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VLLM_BATCH_INVARIANT=1 HF_HUB_OFFLINE=1
extra=()
if [ "$FAMILY" = q8 ]; then extra+=(--target-path /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target); fi
"$study_python" -m experiments.auf_vllm.prepare_family_benchmark --family "$FAMILY" --models "$study/models" "${extra[@]}" --manifest "$study/manifests/dev.jsonl" --out "$run/evaluation"
export_args=()
case "$MODE" in
  ar) ;;
  zip) export_args+=(--export "$study/fits/$FAMILY-n4096-zip-lr1e-3-s42/epoch-3/export");;
  handoff-r56|handoff-five)
    kind=fusion_r56
    if [ "$MODE" = handoff-five ]; then kind=five_maps; fi
    "$study_python" -c "import json;assert json.load(open('$run/modules/full-$kind/$kind/verification.json'))['status']=='passed'"
    export_args+=(--export "$run/exports/full-$kind/$kind");;
  *) exit 2;;
esac
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$run/logs/eval-gpu-$MODE-$SLURM_JOB_ID.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
for repeat in 0 1 2; do
  "$study_python" -u -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --models "$study/models" "${extra[@]}" "${export_args[@]}" --data "$run/evaluation" --mode "$MODE" --count 128 --cap 2048 --repeat "$repeat" --out "$run/measurements"
done
