#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --nodelist=node07
#SBATCH --time=08:00:00
#SBATCH --job-name=handoff-large-fit
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${RECORDS:?Set distinct training records}"
UPDATES=$(( RECORDS / 2 ))
[[ "$RECORDS" =~ ^[0-9]+$ && "$UPDATES" =~ ^[0-9]+$ ]]
(( RECORDS >= 4096 && RECORDS <= 32768 && RECORDS % 4096 == 0 && UPDATES > 0 ))
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
snapshot="$root/scaling-source-$SLURM_JOB_ID"
mkdir -p "$snapshot/experiments"
rsync -a --exclude overlay --exclude specforge --exclude reports --exclude __pycache__ "$control/experiments/handoff_transfer/" "$snapshot/experiments/handoff_transfer/"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm/" "$snapshot/experiments/auf_vllm/"
code="$snapshot/experiments/handoff_transfer"
export PYTHONPATH="$snapshot"
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
fit="$root/large-initializers/q8-n$RECORDS-s42"
data_name=main-q8-n16384
if [ "$RECORDS" = 32768 ]; then
  data_name=main-q8-n32768
  export TRANSFER_WORK="$study/$data_name"
fi
"$study_python" -u -m experiments.auf_vllm.train_zip --records "$RECORDS" --epochs 3 --seed 42 --lr 1e-3 --manifest "$study/$data_name/train.json" --out "$fit"
# Job-specific outputs preserve failed-run diagnostics and their export hashes.
export AUF_WORKDIR="$root/scaling/q8-n$RECORDS-u$UPDATES-$SLURM_JOB_ID"
"$study_python" -m experiments.handoff_transfer.prepare_large --study "$study" --data-name "$data_name" --fit "$fit" --out "$AUF_WORKDIR" --records "$RECORDS"
export SPECFORGE_ROOT="$control/experiments/handoff_transfer/specforge"
export AUF_MODELS="$root/models/qwen4" AUF_DOMAIN=math
export PYTHONPATH="$code/port:$code/vendor/code:$control/experiments/handoff_transfer/overlay:$SPECFORGE_ROOT:$snapshot"
epochs=$(( (UPDATES * 8 + RECORDS - 1) / RECORDS ))
mkdir -p "$AUF_WORKDIR/logs"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$AUF_WORKDIR/logs/gpu.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
"$study_python" "$code/vendor/code/check_loss.py"
for kind in fusion_r56 five_maps; do
  "$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$AUF_WORKDIR/scaling_train.py" --kind "$kind" --tag full --output-tag "full-$kind" --epochs "$epochs" --max-steps "$UPDATES" --snapshot-every "$(( RECORDS / 8 ))"
  "$study_python" "$code/verify_transfer.py" --kind "$kind" --tag "full-$kind" --steps "$UPDATES"
done
# Full128/cap2048 decoding evaluations run separately after fit verification.
echo "SCALING_FIT_VERIFIED $AUF_WORKDIR"
