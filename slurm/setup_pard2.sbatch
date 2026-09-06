#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --nodelist=node07
#SBATCH --job-name=rs-pard2-setup
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${RELAYSPEC_PROJECT_DIR:?}"
export TRANSFORMERS_CACHE="/scratch/$USER/factorspec-runtime-20260826/huggingface/hub"
export RELAYSPEC_OUTPUT="$PWD/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RELAYSPEC_OUTPUT"
export PYTHONPATH="/scratch/$USER/factorspec-runtime-20260826/pard-python-overlay-27823:$PWD/src"
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
"$HOME/.venvs/factorspec-20260826/bin/python" scripts/setup_pard2_checkpoint.py --config configs/submission/competitors-length-20260906/pard2-pilot.json
