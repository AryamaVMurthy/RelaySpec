#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=matrix-transformers
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
FAMILY=q8
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
run=/scratch/aryama.murthy/handoff-transfer-20260911/$FAMILY
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
out="$run/matrix-transformers-$SLURM_JOB_ID"
mkdir -p "$out/source/experiments" "$out/source/src"
rsync -a --exclude reports --exclude overlay --exclude specforge --exclude __pycache__ "$control/experiments/handoff_transfer/" "$out/source/experiments/handoff_transfer/"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm/" "$out/source/experiments/auf_vllm/"
rsync -a --exclude __pycache__ "$control/src/relayspec/" "$out/source/src/relayspec/"
export PYTHONPATH="$out/source:$out/source/src"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
target="$study/models/llama3-target"
if [ "$FAMILY" = q8 ]; then target=/scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target; fi
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$out/gpu.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
"$study_python" -u -m experiments.handoff_transfer.matrix.transformers_final --root /scratch/aryama.murthy/handoff-transfer-20260911 --target "$target" --dflash-source /home/aryama.murthy/factorspec/vendor/dflash --out "$out"
