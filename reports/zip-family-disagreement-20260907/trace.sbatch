#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --nodelist=node07
#SBATCH --time=00:10:00
#SBATCH --job-name=rs-zip-trace
#SBATCH --output=/home/aryama.murthy/rs-zip-family-20260907/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/rs-zip-family-20260907/logs/%x-%j.err
set -euo pipefail
module load u22/cuda/12.4
cd /home/aryama.murthy/rs-zip-family-20260907
export HF_HOME=/scratch/aryama.murthy/factorspec-runtime-20260826/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export DFLASH_SOURCE="$HOME/factorspec/vendor/dflash"
export PYTHONPATH="$PWD/src:$DFLASH_SOURCE"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export ZIP_FAMILY_OUTPUT=/scratch/aryama.murthy/zip-family-pilot-20260907
PY="$HOME/.venvs/factorspec-20260826/bin/python"
mkdir -p "outputs/$SLURM_JOB_ID"
out="$PWD/outputs/$SLURM_JOB_ID"
sha256sum scripts/train_zip_family_pilot.py scripts/benchmark_relay.py external/transfer-reproduction/transfer/src/mapper.py src/relayspec/*.py > "$out/source-sha256.txt"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used --format=csv -l 15 > "$out/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
export TRACE_OUTPUT="$out"
sha256sum scripts/trace_zip_family_disagreement.py > "$out/trace-source-sha256.txt"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/trace_zip_family_disagreement.py
date --iso-8601=seconds > "$out/COMPLETE"
