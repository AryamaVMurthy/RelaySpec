#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --nodelist=node07
#SBATCH --job-name=rs-pard-setup
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${RELAYSPEC_PROJECT_DIR:?immutable source required}"
export RELAYSPEC_CACHE_DIR="/scratch/$USER/factorspec-runtime-20260826"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export RELAYSPEC_OUTPUT="$PWD/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
export PARD_OVERLAY="$RELAYSPEC_CACHE_DIR/pard-python-overlay-$SLURM_JOB_ID"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
mkdir -p "$RELAYSPEC_OUTPUT"
cp SOURCE_COMMIT "$RELAYSPEC_OUTPUT/source-commit.txt"
"$HOME/.local/bin/uv" pip install --python "$HOME/.venvs/factorspec-20260826/bin/python" \
  --target "$PARD_OVERLAY" 'transformers==4.51.3' 'numpy==1.26.4' 'loguru==0.7.3' 'click==8.1.8'
export PYTHONPATH="$PARD_OVERLAY:$PWD/src"
"$HOME/.venvs/factorspec-20260826/bin/python" scripts/setup_pard_checkpoint.py \
  --config configs/submission/baselines/pard-pilot.yaml
