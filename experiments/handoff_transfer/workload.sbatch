#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=handoff-workload128
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
: "${FAMILY:?Set q8, q14, or llama}"
study_snapshot="outputs/source-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$study_snapshot/experiments"
cp -r experiments/auf_vllm "$study_snapshot/experiments/"
export PYTHONPATH="$PWD/$study_snapshot/experiments/auf_vllm/runtime_zip:$PWD/$study_snapshot"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
study_root=/scratch/aryama.murthy/relayspec-auf-20260911
workloads=(math gsm code chat)
workload=${workloads[$((SLURM_ARRAY_TASK_ID / 3))]}
repeat=$((SLURM_ARRAY_TASK_ID % 3))
data="$study_root/benchmarks/workloads/${FAMILY}/${workload}/development"
run="/scratch/aryama.murthy/handoff-transfer-20260911/$FAMILY"
out="$run/workloads/$workload/development"
target_args=()
if [ "$FAMILY" = q8 ]; then
  target_args+=(--target-path /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target)
fi
cd "$study_snapshot"
"$study_python" -m experiments.auf_vllm.prepare_workload_benchmark --manifest "$study_root/workloads-20260911" --family "$FAMILY" --workload "$workload" --split development --out "$data"
modes=(ar normal zip handoff-r56 handoff-five)
if [ "$FAMILY" = q8 ]; then modes+=(native); fi
count=128
cap=2048
if [ "${GATE_ONLY:-0}" = 1 ]; then count=4;cap=128;out="$out/diagnostic";fi
for ((i=0; i<${#modes[@]}; i++)); do
  mode=${modes[$(((i+repeat) % ${#modes[@]}))]}
  extra=()
  if [ "$mode" = native ]; then
    extra+=(--native-draft /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/draft)
  elif [ "$mode" != ar ]; then
    case "$mode" in
      normal) extra+=(--export "$run/normal/export");;
      zip) extra+=(--export "$study_root/fits/${FAMILY}-n4096-zip-lr1e-3-s42/epoch-3/export");;
      handoff-r56) extra+=(--export "$run/exports/full-fusion_r56/fusion_r56");;
      handoff-five) extra+=(--export "$run/exports/full-five_maps/five_maps");;
    esac
  fi
  "$study_python" -u -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --models "$study_root/models" "${target_args[@]}" --data "$data" --mode "$mode" "${extra[@]}" --count "$count" --cap "$cap" --repeat "$repeat" --out "$out"
done
for mode in "${modes[@]}"; do
  if [ "$mode" = ar ]; then continue;fi
  extra=()
  extra+=(--require-exact)
  "$study_python" -m experiments.auf_vllm.compare_outputs --ar "$out/ar-r${repeat}-w0.jsonl" --method "$out/${mode}-r${repeat}-w0.jsonl" --out "$out/${mode}-comparison-r${repeat}.json" --expected-count "$count" "${extra[@]}"
done
