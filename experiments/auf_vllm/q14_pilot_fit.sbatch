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
#SBATCH --job-name=auf-q14-pilot-fit
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VLLM_BATCH_INVARIANT=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/experiments/auf_vllm/bootstrap:$PWD"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
study_out="/scratch/aryama.murthy/relayspec-auf-20260911/pilot-q14-fit-${SLURM_JOB_ID}"
for loss in auf ce zip; do
  "$study_python" -u -m experiments.auf_vllm.train_llama_pilot --family q14 --objective "$loss" --out "$study_out/$loss"
done
