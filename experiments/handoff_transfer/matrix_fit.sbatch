#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --job-name=handoff-matrix
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${FAMILY:?}" "${KIND:?}" "${MATRIX_OBJECTIVE:?}" "${MATRIX_LR:?}"
control=/home/aryama.murthy/relayspec-auf-20260911
code="$control/experiments/handoff_transfer"
root=/scratch/aryama.murthy/handoff-transfer-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export AUF_WORKDIR="$root/matrix/$FAMILY/$KIND-$MATRIX_OBJECTIVE-lr$MATRIX_LR"
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
if [ "${TRAIN_STEPS:-0}" -gt 100 ]; then
 # A promoted fit has already passed a trained-checkpoint decoding screen.
 # Revalidate those artifacts instead of repeating the two-update smoke run.
 prior_steps=100
 if [ "$TRAIN_STEPS" -eq 2000 ]; then prior_steps=500; fi
 "$study_python" -m experiments.handoff_transfer.matrix.select --runs "$AUF_WORKDIR" --steps "$prior_steps" --out "$AUF_WORKDIR/promotion-preflight-$SLURM_JOB_ID.json"
else
"$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$matrix_code/train.py" --kind "$KIND" --tag check --output-tag "gate-$SLURM_JOB_ID" --lr "$MATRIX_LR" --epochs 1 --max-steps 2
"$study_python" "$matrix_code/verify.py" --kind "$KIND" --tag "gate-$SLURM_JOB_ID" --steps 2
# Check the actual deployed decoder before any tuning fit.
training_gpus="$CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" VLLM_BATCH_INVARIANT=1
study=/scratch/aryama.murthy/relayspec-auf-20260911
extra=()
if [ "$FAMILY" = q8 ]; then extra+=(--target-path /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target); fi
for mode in ar matrix; do
 export_args=()
 if [ "$mode" = matrix ]; then export_args+=(--export "$AUF_WORKDIR/exports/gate-$SLURM_JOB_ID/$KIND"); fi
 "$study_python" -u -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --models "$study/models" "${extra[@]}" "${export_args[@]}" --data "$root/$FAMILY/evaluation" --mode "$mode" --count 4 --cap 128 --out "$AUF_WORKDIR/gate-$SLURM_JOB_ID"
done
"$study_python" -m experiments.auf_vllm.compare_outputs --ar "$AUF_WORKDIR/gate-$SLURM_JOB_ID/ar-r0-w0.jsonl" --method "$AUF_WORKDIR/gate-$SLURM_JOB_ID/matrix-r0-w0.jsonl" --out "$AUF_WORKDIR/gate-$SLURM_JOB_ID/exactness.json" --require-exact --expected-count 4
export CUDA_VISIBLE_DEVICES="$training_gpus"
fi
# TRAIN_STEPS defaults to gate only. Tuning screens are explicit.

if [ "${TRAIN_STEPS:-0}" -gt 0 ]; then
 "$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$matrix_code/train.py" --kind "$KIND" --tag full --output-tag "steps-$TRAIN_STEPS" --lr "$MATRIX_LR" --epochs 4 --max-steps "$TRAIN_STEPS" --snapshot-every 100
 "$study_python" "$matrix_code/verify.py" --kind "$KIND" --tag "steps-$TRAIN_STEPS" --steps "$TRAIN_STEPS"
fi
