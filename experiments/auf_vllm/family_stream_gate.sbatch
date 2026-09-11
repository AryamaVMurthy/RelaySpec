#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node06
#SBATCH --time=00:15:00
#SBATCH --job-name=auf-family-stream-gate
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
for family in llama q14; do
  if [ "$family" = llama ]; then data="$study_root/pilot-l3"; else data="$study_root/pilot-q14"; fi
  for loss in auf ce zip; do
    "$study_python" -u -m experiments.auf_vllm.family_training --family "$family" --models "$study_root/models" --data "$data" --validation-manifest "$data/dev.json" --records 32 --epochs 1 --objective "$loss" --exploratory --out "$study_root/stream-gate-${SLURM_JOB_ID}/$family/$loss"
  done
done
