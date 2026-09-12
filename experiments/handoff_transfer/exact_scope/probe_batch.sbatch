#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911
code="$control/experiments/handoff_transfer"
root=/scratch/aryama.murthy/handoff-transfer-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export MATRIX_NUM_ANCHORS=32 AUF_WORKDIR="$root/batch-probes/$FAMILY-$KIND-$MATRIX_OBJECTIVE-$SLURM_JOB_ID" AUF_MODELS="$root/models/qwen4" AUF_DOMAIN=math SPECFORGE_ROOT="$code/specforge"
export PYTHONPATH="$control:$code/matrix:$code/overlay:$code/vendor/code:$SPECFORGE_ROOT"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
"$study_python" -m experiments.handoff_transfer.matrix.prepare --base "$root/$FAMILY" --out "$AUF_WORKDIR" --kind "$KIND" --objective "$MATRIX_OBJECTIVE" --lr .0001
"$study_python" "$code/matrix/probe_batch_cli.py" --kind "$KIND"
