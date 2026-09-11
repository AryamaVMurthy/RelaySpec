#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --job-name=handoff-cross-fit
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911
code="$control/experiments/handoff_transfer"
root=/scratch/aryama.murthy/handoff-transfer-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export PYTHONPATH="$control:$code/matrix:$code/overlay:$code/vendor/code:$code/specforge"
export AUF_WORKDIR="$root/anchors512/cross/pilot16" AUF_MODELS="$root/models/qwen4" AUF_DOMAIN=math SPECFORGE_ROOT="$code/specforge"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
"$study_python" -m experiments.handoff_transfer.cross.initializer --archived "$root/archived-cross-pilot" --reference "$root/q8" --out "$AUF_WORKDIR"
for objective in ce auf; do
 export MATRIX_OBJECTIVE="$objective"
 "$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$code/cross/train.py" --kind five_maps --steps 2
 done
