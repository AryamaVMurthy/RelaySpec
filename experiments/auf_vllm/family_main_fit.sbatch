#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node06
#SBATCH --time=02:00:00
#SBATCH --array=0-2%1
#SBATCH --job-name=auf-family-mainfit
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VLLM_BATCH_INVARIANT=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/experiments/auf_vllm/bootstrap:$PWD"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
study_root=/scratch/aryama.murthy/relayspec-auf-20260911
study_snapshot="outputs/source-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$study_snapshot/experiments"
cp -r experiments/auf_vllm "$study_snapshot/experiments/"
export PYTHONPATH="$PWD/$study_snapshot"
cd "$study_snapshot"
: "${FAMILY:?Set llama or q14}"
losses=(zip ce auf)
loss=${losses[$SLURM_ARRAY_TASK_ID]}
if [ "$FAMILY" = llama ]; then data="$study_root/main-l3-n4096"; else data="$study_root/main-q14-n4096"; fi
valid="$study_root/validation-${FAMILY}-n1024"
manifest="$valid/combined-${loss}.json"
"$study_python" -m experiments.auf_vllm.family_offline --data "$valid" --manifest-only --out "$manifest"
# Transfer the frozen Q8 LR choice across same-family targets; do not tune it
# on these families' decoding sets.
rate=$("$study_python" -m experiments.auf_vllm.select_lr --root /home/aryama.murthy/relayspec-auf-20260911/outputs/lr-screen --out "$study_root/lr-selection-${FAMILY}-${loss}.json" --loss "$loss")
out="$study_root/fits/${FAMILY}-n4096-${loss}-lr${rate}-s42"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 1 > "/home/aryama.murthy/relayspec-auf-20260911/logs/gpu-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
"$study_python" -u -m experiments.auf_vllm.family_training --family "$FAMILY" --models "$study_root/models" --data "$data" --validation-manifest "$manifest" --records 4096 --epochs 3 --objective "$loss" --lr "$rate" --out "$out"
"$study_python" -u -m experiments.auf_vllm.family_offline --family "$FAMILY" --models "$study_root/models" --data "$valid" --fit "$out"
