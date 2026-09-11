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
#SBATCH --job-name=handoff-ce-control
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer
base=/scratch/aryama.murthy/handoff-transfer-20260911
run="$base/q8-ce"
mkdir -p "$run" "$run/logs"
ln -sfn "$base/q8/transfer.json" "$run/transfer.json"
ln -sfn "$base/q8/features" "$run/features"
export SPECFORGE_ROOT="$control/specforge" AUF_WORKDIR="$run" AUF_MODELS="$base/models/qwen4" AUF_DOMAIN=math
export PYTHONPATH="$control/overlay:$control/port_ce:$control/port:$control/vendor/code:$SPECFORGE_ROOT"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$run/logs/gpu-$SLURM_JOB_ID.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
"$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$control/port_ce/train.py" --kind fusion_r56 --tag check --output-tag check-fusion_r56 --epochs 1 --max-steps 2
"$study_python" "$control/verify_transfer.py" --kind fusion_r56 --tag check-fusion_r56 --steps 2
"$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$control/port_ce/train.py" --kind fusion_r56 --tag full --output-tag full-fusion_r56 --epochs 4 --max-steps 2000
"$study_python" "$control/verify_transfer.py" --kind fusion_r56 --tag full-fusion_r56 --steps 2000
