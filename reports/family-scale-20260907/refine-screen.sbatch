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
#SBATCH --job-name=rs-family-scale
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
export FAMILY_SCALE_CACHE=/scratch/aryama.murthy/family-scale-20260907
export FAMILY_SCALE_DATA=/scratch/aryama.murthy/factorspec-runtime-20260826/relayspec/scaling-data/numina-v1
pids=()
IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
for lane in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${devices[$lane]}" "$PY" scripts/run_family_candidates.py --list "$FAMILY_SCALE_CACHE/refinement/screen-lane${lane}.json" --output "$FAMILY_SCALE_CACHE/refine-screen/lane$lane" > "$FAMILY_SCALE_CACHE/refine-screen-lane$lane.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
