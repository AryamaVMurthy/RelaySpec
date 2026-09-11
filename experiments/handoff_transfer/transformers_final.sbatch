#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=handoff-transformers
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${FAMILY:?Set q8 or llama}"
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
run=/scratch/aryama.murthy/handoff-transfer-20260911/$FAMILY
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
out="$run/transformers-$SLURM_JOB_ID"
mkdir -p "$out/source/experiments" "$out/source/src"
rsync -a --exclude reports --exclude overlay --exclude specforge --exclude __pycache__ "$control/experiments/handoff_transfer/" "$out/source/experiments/handoff_transfer/"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm/" "$out/source/experiments/auf_vllm/"
rsync -a --exclude __pycache__ "$control/src/relayspec/" "$out/source/src/relayspec/"
export PYTHONPATH="$out/source:$out/source/src"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
target="$study/models/llama3-target"
if [ "$FAMILY" = q8 ]; then target=/scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target; fi
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$out/gpu.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
for phase in gate full; do
  count=4;cap=128
  if [ "$phase" = full ]; then count=128;cap=2048;fi
  for mode in ar normal zip handoff-r56 handoff-five; do
    extra=()
    case "$mode" in
      ar) ;;
      normal) extra+=(--export "$run/normal/export");;
      zip) extra+=(--export "$study/fits/$FAMILY-n4096-zip-lr1e-3-s42/epoch-3/export");;
      handoff-r56) extra+=(--export "$run/exports/full-fusion_r56/fusion_r56");;
      handoff-five) extra+=(--export "$run/exports/full-five_maps/five_maps");;
    esac
    "$study_python" -u -m experiments.handoff_transfer.transformers_benchmark --family "$FAMILY" --target "$target" --data "$run/evaluation" --out "$out/$phase" --dflash-source /home/aryama.murthy/factorspec/vendor/dflash --mode "$mode" "${extra[@]}" --count "$count" --cap "$cap"
    if [ "$mode" != ar ]; then
      "$study_python" -m experiments.auf_vllm.compare_outputs --ar "$out/$phase/ar-r0-w0.jsonl" --method "$out/$phase/$mode-r0-w0.jsonl" --out "$out/$phase/$mode-comparison.json" --require-exact --expected-count "$count"
    fi
  done
done
