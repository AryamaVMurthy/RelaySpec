#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node06
#SBATCH --time=00:10:00
#SBATCH --job-name=auf-q14-preflight
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VLLM_BATCH_INVARIANT=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/experiments/auf_vllm/bootstrap:$PWD"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
"$study_python" -u -m experiments.auf_vllm.family_preflight generate --family q14 --out "outputs/q14-generate-${SLURM_JOB_ID}.json"
AUF_CAPTURE=1 AUF_TARGET_TAPS='[1,10,19,28,37]' "$study_python" -u -m experiments.auf_vllm.family_preflight capture --family q14 --out "outputs/q14-capture-${SLURM_JOB_ID}.json"
