#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --job-name=handoff-transfer
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${FAMILY:?Set family}" "${KIND:?Set fusion_r56 or five_maps}"
study_control=/home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer
study_root=/scratch/aryama.murthy/handoff-transfer-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export SPECFORGE_ROOT="$study_control/specforge"
export PYTHONPATH="$study_control/overlay:$study_control/port:$study_control/vendor/code:$SPECFORGE_ROOT"
export AUF_MODELS="$study_root/models/qwen4" AUF_WORKDIR="$study_root/$FAMILY" AUF_DOMAIN=math
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p "$AUF_WORKDIR/features/check" "$AUF_WORKDIR/logs"
ln -sfn ../full/00000.pt "$AUF_WORKDIR/features/check/00000.pt"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$AUF_WORKDIR/logs/gpu-$KIND-$SLURM_JOB_ID.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid"' EXIT
"$study_python" "$study_control/vendor/code/check_loss.py"
"$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$study_control/port/train.py" --kind "$KIND" --tag check --output-tag "check-$KIND" --epochs 1 --max-steps 2
"$study_python" "$study_control/verify_transfer.py" --kind "$KIND" --tag "check-$KIND" --steps 2
if [ "${RUN_FULL:-0}" = 1 ]; then
  "$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$study_control/port/train.py" --kind "$KIND" --tag full --output-tag "full-$KIND" --epochs 4 --max-steps 2000
  "$study_python" "$study_control/verify_transfer.py" --kind "$KIND" --tag "full-$KIND" --steps 2000
fi
