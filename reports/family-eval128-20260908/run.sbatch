#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --nodelist=node07
#SBATCH --time=00:15:00
#SBATCH --array=0-15%1
#SBATCH --job-name=rs-family128
#SBATCH --output=/home/aryama.murthy/rs-zip-family-20260907/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/rs-zip-family-20260907/logs/%x-%A_%a.err
set -euo pipefail
module load u22/cuda/12.4
cd /home/aryama.murthy/rs-zip-family-20260907
export HF_HOME=/scratch/aryama.murthy/factorspec-runtime-20260826/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
export PYTHONPATH="$PWD/src:$DFLASH_SOURCE"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
PY=/home/aryama.murthy/.venvs/factorspec-20260826/bin/python
OUT=/scratch/aryama.murthy/family-eval128-20260908/wave$SLURM_ARRAY_TASK_ID
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv > "$OUT/allocation.csv"
pids=()
IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
for lane in 0 1 2 3; do
 CUDA_VISIBLE_DEVICES="${devices[$lane]}" "$PY" scripts/run_family_shared_candidates.py --list "reports/family-eval128-20260908/wave${SLURM_ARRAY_TASK_ID}-lane${lane}.json" --output "$OUT/lane$lane" > "$OUT/lane$lane.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
