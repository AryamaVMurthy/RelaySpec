#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --time=00:16:00
#SBATCH --nodelist=node07
#SBATCH --job-name=native-joint
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module load u22/cuda/12.4
: "${WAVE:?Specify a wave config}"
export DFLASH_SOURCE="$HOME/factorspec/vendor/dflash"
export HF_HOME="/scratch/$USER/factorspec-runtime-20260826/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export NATIVE_SCRATCH="/scratch/$USER/native-joint"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD:$DFLASH_SOURCE"
mkdir -p metadata
sha256sum *.py *.sbatch configs/*.json > "metadata/$SLURM_JOB_ID-source-sha256.txt"
tar -czf "metadata/$SLURM_JOB_ID-source.tar.gz" *.py *.sbatch configs README.md tests
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used --format=csv -l 10 > "metadata/$SLURM_JOB_ID-gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
"$HOME/.venvs/factorspec-20260826/bin/python" run_wave.py
