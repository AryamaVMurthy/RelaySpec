#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=exact-feature-fit
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
snapshot=/scratch/aryama.murthy/exact32-source-$SLURM_JOB_ID
mkdir -p "$snapshot/experiments" "$snapshot/src"
rsync -a --exclude reports --exclude __pycache__ --exclude overlay --exclude specforge --exclude vendor --exclude data "$control/experiments/auf_vllm" "$control/experiments/handoff_transfer" "$snapshot/experiments/"
rsync -a --exclude __pycache__ "$control/src/relayspec" "$snapshot/src/"
export PYTHONPATH="$snapshot/src:$snapshot" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$snapshot/gpu.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT

export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
study=/scratch/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
base="$root/$FAMILY"
case "$FAMILY" in
 q8) target="$TRANSFER_WORK/models/8b/target";;
 llama) target="$study/models/llama3-target";;
 cross) target="$study/models/llama8-source";base="$root/cross-full4096/initializers-e1";;
 *) exit 2;;
esac
"$study_python" -m experiments.handoff_transfer.train_normal --family "$FAMILY" --root "$base" --study "$study" --target "$target" --epochs 1 --seed 42 --lr 1e-3 --objective "$MATRIX_OBJECTIVE" --out "$root/feature-objectives-e1/$FAMILY/$MATRIX_OBJECTIVE"
