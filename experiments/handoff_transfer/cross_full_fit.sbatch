#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --job-name=cross-full-fit
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${KIND:?}" "${MATRIX_OBJECTIVE:?}" "${MATRIX_LR:?}" "${TRAIN_STEPS:?}" "${CROSS_INDEX:?}"
control=/home/aryama.murthy/relayspec-auf-20260911
code="$control/experiments/handoff_transfer"
root=/scratch/aryama.murthy/handoff-transfer-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export AUF_WORKDIR="$root/matrix/cross/$KIND-$MATRIX_OBJECTIVE-lr$MATRIX_LR"
export AUF_MODELS="$root/cross-reference" AUF_DOMAIN=math SPECFORGE_ROOT="$code/specforge"
snapshot="$AUF_WORKDIR/source-$SLURM_JOB_ID"
mkdir -p "$snapshot/experiments/handoff_transfer" "$AUF_WORKDIR/logs"
rsync -a --exclude __pycache__ "$code/matrix" "$code/cross" "$snapshot/experiments/handoff_transfer/"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm" "$snapshot/experiments/"
export PYTHONPATH="$snapshot:$snapshot/experiments/handoff_transfer/matrix:$code/overlay:$code/vendor/code:$SPECFORGE_ROOT"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
"$study_python" -m experiments.handoff_transfer.cross.prepare_cell --initializers "$root/cross-full4096/initializers" --index "$CROSS_INDEX" --out "$AUF_WORKDIR" --kind "$KIND" --objective "$MATRIX_OBJECTIVE" --lr "$MATRIX_LR"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw --format=csv -l 5 > "$AUF_WORKDIR/logs/gpu-$SLURM_JOB_ID.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
"$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$snapshot/experiments/handoff_transfer/cross/full_train.py" --index "$CROSS_INDEX" --kind "$KIND" --steps "$TRAIN_STEPS" --lr "$MATRIX_LR"
