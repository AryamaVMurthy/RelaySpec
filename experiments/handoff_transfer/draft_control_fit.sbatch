#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodelist=node07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --job-name=handoff-draft-control
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${AUF_DRAFT_RANK:?Set rank}"
control=/home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer
base=/scratch/aryama.murthy/handoff-transfer-20260911
run="$base/q8-draft-control-r$AUF_DRAFT_RANK"
mkdir -p "$run" "$run/logs"
ln -sfn "$base/q8/transfer.json" "$run/transfer.json"
ln -sfn "$base/q8/features" "$run/features"
export SPECFORGE_ROOT="$control/specforge" AUF_WORKDIR="$run" AUF_MODELS="$base/models/qwen4" AUF_DOMAIN=math
export PYTHONPATH="$control/overlay:$control/draft_control:$control/port:$control/vendor/code:$SPECFORGE_ROOT"
export PYTHONPATH="/home/aryama.murthy/relayspec-auf-20260911:$PYTHONPATH"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$run/logs/gpu-$SLURM_JOB_ID.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
"$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$control/draft_control/train.py" --kind draft_lora --tag check --output-tag check-draft_lora-$SLURM_JOB_ID --epochs 1 --max-steps 2
"$study_python" "$control/draft_control/verify.py" --tag check-draft_lora-$SLURM_JOB_ID --steps 2
"$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$control/draft_control/train.py" --kind draft_lora --tag full --output-tag full-draft_lora --epochs 4 --max-steps 2000
"$study_python" "$control/draft_control/verify.py" --tag full-draft_lora --steps 2000
