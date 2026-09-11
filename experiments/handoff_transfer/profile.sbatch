#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --job-name=handoff-gpu-profile
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
: "${FAMILY:?Set q8 or llama}"
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
run=/scratch/aryama.murthy/handoff-transfer-20260911/$FAMILY
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
modes=(ar normal zip handoff-r56 handoff-five)
mode=${modes[$SLURM_ARRAY_TASK_ID]}
out="$run/profiles/$mode-$SLURM_ARRAY_JOB_ID"
mkdir -p "$out/source/experiments"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm/" "$out/source/experiments/auf_vllm/"
export PYTHONPATH="$out/source"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
extra=()
if [ "$FAMILY" = q8 ]; then extra+=(--target-path /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target); fi
case "$mode" in
  ar) ;;
  normal) extra+=(--export "$run/normal/export");;
  zip) extra+=(--export "$study/fits/$FAMILY-n4096-zip-lr1e-3-s42/epoch-3/export");;
  handoff-r56) extra+=(--export "$run/exports/full-fusion_r56/fusion_r56");;
  handoff-five) extra+=(--export "$run/exports/full-five_maps/five_maps");;
esac
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 1 > "$out/gpu.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
# Instrumentation is isolated from primary timing. The profiler records at
# most64 engine iterations; these traces do not measure full-run SM occupancy.
"$study_python" -u -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --models "$study/models" "${extra[@]}" --data "$run/evaluation" --mode "$mode" --count 4 --cap 2048 --out "$out/requests" --profile-dir "$out/traces"
