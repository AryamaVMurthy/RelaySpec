#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node06
#SBATCH --time=06:00:00
#SBATCH --job-name=auf-family-dev128
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VLLM_BATCH_INVARIANT=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/experiments/auf_vllm/bootstrap:$PWD"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
study_root=/scratch/aryama.murthy/relayspec-auf-20260911
study_snapshot="outputs/source-${SLURM_JOB_ID}"
mkdir -p "$study_snapshot/experiments"
cp -r experiments/auf_vllm "$study_snapshot/experiments/"
export PYTHONPATH="$PWD/$study_snapshot"
cd "$study_snapshot"
: "${FAMILY:?Set llama or q14}"
models="$study_root/models"
data="$study_root/benchmarks/${FAMILY}-development"
"$study_python" -m experiments.auf_vllm.prepare_family_benchmark --family "$FAMILY" --models "$models" --manifest "$study_root/manifests/dev.jsonl" --out "$data"
export VLLM_BATCH_INVARIANT=1
out="$study_root/benchmarks/${FAMILY}-selected"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 1 > "/home/aryama.murthy/relayspec-auf-20260911/logs/gpu-${SLURM_JOB_ID}.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
# Verify the same telemetry/asset path and short exactness before128 full runs.
"$study_python" -u -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --models "$models" --data "$data" --mode ar --count 4 --cap 128 --out "$out/diagnostic"
for loss in zip ce auf; do
  rate=$("$study_python" -m experiments.auf_vllm.select_lr --root /home/aryama.murthy/relayspec-auf-20260911/outputs/lr-screen --out "$study_root/lr-selection-${FAMILY}-decode.json" --loss "$loss")
  fit="$study_root/fits/${FAMILY}-n4096-${loss}-lr${rate}-s42/epoch-3/export"
  "$study_python" -u -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --models "$models" --data "$data" --mode "$loss" --export "$fit" --count 4 --cap 128 --out "$out/diagnostic"
  "$study_python" -m experiments.auf_vllm.compare_outputs --ar "$out/diagnostic/ar-r0-w0.jsonl" --method "$out/diagnostic/${loss}-r0-w0.jsonl" --out "$out/diagnostic/${loss}-exactness.json" --require-exact --expected-count 4
done
"$study_python" -u -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --models "$models" --data "$data" --mode ar --count 128 --cap 2048 --out "$out/development128-cap2048"
for loss in zip ce auf; do
  rate=$("$study_python" -m experiments.auf_vllm.select_lr --root /home/aryama.murthy/relayspec-auf-20260911/outputs/lr-screen --out "$study_root/lr-selection-${FAMILY}-decode.json" --loss "$loss")
  fit="$study_root/fits/${FAMILY}-n4096-${loss}-lr${rate}-s42/epoch-3/export"
  "$study_python" -u -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --models "$models" --data "$data" --mode "$loss" --export "$fit" --count 128 --cap 2048 --out "$out/development128-cap2048"
  "$study_python" -m experiments.auf_vllm.compare_outputs --ar "$out/development128-cap2048/ar-r0-w0.jsonl" --method "$out/development128-cap2048/${loss}-r0-w0.jsonl" --out "$out/development128-cap2048/${loss}-comparison.json" --expected-count 128
done
