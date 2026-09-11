#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node07
#SBATCH --time=08:00:00
#SBATCH --job-name=auf-matched-epochs
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
study_snapshot="outputs/source-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$study_snapshot/experiments"
cp -r experiments/auf_vllm "$study_snapshot/experiments/"
export PYTHONPATH="$PWD/$study_snapshot/experiments/auf_vllm/runtime_zip:$PWD/$study_snapshot"
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
# 4096 main fits already exist; 32768 main fits are queued separately.
counts=(16 32 64 128 256 512 1024 2048 8192 16384)
losses=(zip ce auf)
records=${counts[$((SLURM_ARRAY_TASK_ID / 3))]}
loss=${losses[$((SLURM_ARRAY_TASK_ID % 3))]}
axis=records
if [ "${ZIP_TRAJECTORY:-0}" = 1 ]; then
  counts=(512 4096)
  records=${counts[$SLURM_ARRAY_TASK_ID]}
  loss=zip
  axis=epochs
fi
cd "$study_snapshot"
"$study_python" -u -m experiments.auf_vllm.scaling_cell --root /scratch/aryama.murthy/relayspec-auf-20260911 --axis "$axis" --records "$records" --loss "$loss"
