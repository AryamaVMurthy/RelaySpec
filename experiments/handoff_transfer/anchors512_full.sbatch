#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --job-name=handoff-anchors512-full
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${FAMILY:?Set family}" "${KIND:?Set fusion_r56 or five_maps}"
study_control=/home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer
study_root=/scratch/aryama.murthy/handoff-transfer-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export SPECFORGE_ROOT="$study_control/specforge"
export PYTHONPATH="$study_control/overlay:$study_control/anchors512:$study_control/vendor/code:$SPECFORGE_ROOT"
export AUF_MODELS="$study_root/models/qwen4" AUF_WORKDIR="$study_root/$FAMILY" AUF_DOMAIN=math
# Separate artifacts; share only the immutable cached features and ZIP initializer.
base_workdir="$AUF_WORKDIR"
export AUF_WORKDIR="$study_root/anchors512/$FAMILY"
mkdir -p "$AUF_WORKDIR/features"
cp "$base_workdir/transfer.json" "$AUF_WORKDIR/transfer.json"
ln -sfn "$base_workdir/features/full" "$AUF_WORKDIR/features/full"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
# The bounded pilot already exercised training, folding and AR correctness.
: "${GATE_JOB:?Set the successful 512-anchor pilot job ID}"
gate_tag="check-$KIND-$GATE_JOB"
"$study_python" -c "import json;from pathlib import Path;r=Path('$AUF_WORKDIR');s=json.load(open(r/'modules'/'$gate_tag'/'$KIND'/'summary.json'));assert s['optimizer_steps']==2 and s['anchors_per_example']==512;assert json.load(open(r/'modules'/'$gate_tag'/'$KIND'/'verification.json'))['status']=='passed';g=json.load(open(r/'gate-$KIND-$GATE_JOB'/'exactness.json'));assert g['count']==4 and g['exact_matches']==4"
mkdir -p "$AUF_WORKDIR/logs"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$AUF_WORKDIR/logs/gpu-full-$SLURM_JOB_ID.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid"' EXIT
"$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$study_control/anchors512/train.py" --kind "$KIND" --tag full --output-tag "full-$KIND" --epochs 4 --max-steps 2000
"$study_python" "$study_control/verify_transfer.py" --kind "$KIND" --tag "full-$KIND" --steps 2000
