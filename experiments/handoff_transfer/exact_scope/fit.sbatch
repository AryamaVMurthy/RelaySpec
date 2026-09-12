#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --job-name=handoff-matrix
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
export MATRIX_NUM_ANCHORS=32
case "$KIND:$MATRIX_OBJECTIVE" in normal_ce:ce|five_maps:ce|five_maps:auf|dense_fusion:ce|dense_fusion:auf|five_ba56:ce|five_ba56:auf) ;; *) exit 2;; esac
: "${FAMILY:?}" "${KIND:?}" "${MATRIX_OBJECTIVE:?}" "${MATRIX_LR:?}"
control=/home/aryama.murthy/relayspec-auf-20260911
code="$control/experiments/handoff_transfer"
root=/scratch/aryama.murthy/handoff-transfer-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export AUF_WORKDIR="$root/matrix32e1b8/$FAMILY/$KIND-$MATRIX_OBJECTIVE-lr$MATRIX_LR"
export AUF_MODELS="$root/models/qwen4" AUF_DOMAIN=math SPECFORGE_ROOT="$code/specforge"
snapshot="$AUF_WORKDIR/source-$SLURM_JOB_ID"
mkdir -p "$snapshot/experiments/handoff_transfer/matrix" "$snapshot/experiments/auf_vllm"
rsync -a --exclude __pycache__ "$code/matrix/" "$snapshot/experiments/handoff_transfer/matrix/"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm/" "$snapshot/experiments/auf_vllm/"
matrix_code="$snapshot/experiments/handoff_transfer/matrix"
export PYTHONPATH="$snapshot:$matrix_code:$code/overlay:$code/vendor/code:$SPECFORGE_ROOT"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
"$study_python" -m experiments.handoff_transfer.matrix.prepare --base "$root/$FAMILY" --out "$AUF_WORKDIR" --kind "$KIND" --objective "$MATRIX_OBJECTIVE" --lr "$MATRIX_LR"
mkdir -p "$AUF_WORKDIR/logs"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw --format=csv -l 5 > "$AUF_WORKDIR/logs/gpu-$SLURM_JOB_ID.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
"$study_python" -m torch.distributed.run --standalone --nproc_per_node=1 "$matrix_code/train.py" --kind "$KIND" --tag check --output-tag "gate-$SLURM_JOB_ID" --lr "$MATRIX_LR" --epochs 1 --max-steps 2
"$study_python" "$matrix_code/verify.py" --kind "$KIND" --tag "gate-$SLURM_JOB_ID" --steps 2
# The two-update check verifies fit/export integrity. The full128-request
# evaluation checks deployed exactness after the completed one-epoch fit.
if [ "${TRAIN_STEPS:-0}" -gt 0 ]; then
 "$study_python" -m torch.distributed.run --standalone --nproc_per_node=1 "$matrix_code/train.py" --kind "$KIND" --tag full --output-tag "steps-$TRAIN_STEPS" --lr "$MATRIX_LR" --epochs 1 --max-steps "$TRAIN_STEPS" --snapshot-every 500
 "$study_python" "$matrix_code/verify.py" --kind "$KIND" --tag "steps-$TRAIN_STEPS" --steps "$TRAIN_STEPS"
fi
