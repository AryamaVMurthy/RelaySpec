#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=cross-full-data
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
: "${SLURM_ARRAY_TASK_ID:?}"
[ "$SLURM_ARRAY_TASK_ID" -ge 0 ] && [ "$SLURM_ARRAY_TASK_ID" -lt 64 ]
# Full-data collection requires the actual heterogeneous-tokenizer decoder gate.
for objective in ce auf; do
 "$study_python" -c "import json;v=json.load(open('$root/anchors512/cross/pilot16/decode-gate/$objective/exactness.json'));assert v['exact_matches']==v['finish_matches']==v['count']==4"
done
out="$root/cross-full4096/chunks/$(printf '%05d' "$SLURM_ARRAY_TASK_ID")"
mkdir -p "$out"
snapshot="$out/source-$SLURM_JOB_ID"
mkdir -p "$snapshot/experiments/handoff_transfer/cross" "$snapshot/experiments/handoff_transfer/matrix"
rsync -a --exclude __pycache__ "$control/experiments/handoff_transfer/cross/" "$snapshot/experiments/handoff_transfer/cross/"
rsync -a --exclude __pycache__ "$control/experiments/handoff_transfer/matrix/" "$snapshot/experiments/handoff_transfer/matrix/"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm/" "$snapshot/experiments/auf_vllm/"
export PYTHONPATH="$snapshot" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw --format=csv -l 5 > "$out/gpu-$SLURM_JOB_ID.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
"$study_python" -u -m experiments.handoff_transfer.cross.rollout --out "$out" --count 64 --offset "$((SLURM_ARRAY_TASK_ID*64))"
"$study_python" -u -m experiments.handoff_transfer.cross.prepare --out "$out" --models "$study/models"
export PYTHONPATH="$snapshot/experiments/auf_vllm/bootstrap:$snapshot" AUF_CAPTURE=1
"$study_python" -u -m experiments.handoff_transfer.cross.capture --out "$out"
