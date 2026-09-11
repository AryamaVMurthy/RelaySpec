#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --nodelist=node07
#SBATCH --time=01:00:00
#SBATCH --job-name=auf-offline-validation
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
: "${LOSS:?Set auf, ce, or zip}"
: "${LR:?Set learning rate}"
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONPATH="$PWD"
study_root=/scratch/aryama.murthy/relayspec-auf-20260911
/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python -u -m experiments.auf_vllm.evaluate_offline --fit "$study_root/fits/q8-n4096-${LOSS}-lr${LR}-s42" --data "$study_root/validation-q8-n1024"
