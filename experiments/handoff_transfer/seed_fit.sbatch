#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --nodelist=node07
#SBATCH --time=02:00:00
#SBATCH --job-name=handoff-seed-fit
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${FIT_SEED:?Set43 or44}"
[[ "$FIT_SEED" = 43 || "$FIT_SEED" = 44 ]]
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
snapshot="$root/seed-source-$SLURM_JOB_ID"
mkdir -p "$snapshot/experiments" "$snapshot/src"
rsync -a --exclude overlay --exclude specforge --exclude reports --exclude __pycache__ "$control/experiments/handoff_transfer/" "$snapshot/experiments/handoff_transfer/"
rsync -a --exclude reports --exclude __pycache__ "$control/experiments/auf_vllm/" "$snapshot/experiments/auf_vllm/"
rsync -a --exclude __pycache__ "$control/src/relayspec" "$snapshot/src/"
code="$snapshot/experiments/handoff_transfer"
export PYTHONPATH="$snapshot/src:$snapshot"
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
fit="$root/seed-initializers/q8-s$FIT_SEED"
"$study_python" -u -m experiments.auf_vllm.train_zip --records 4096 --epochs 3 --seed "$FIT_SEED" --lr 1e-3 --manifest "$study/main-q8-n4096/train.json" --out "$fit"
export AUF_WORKDIR="$root/seeds/q8-s$FIT_SEED"
"$study_python" -m experiments.handoff_transfer.prepare_seed --parent "$root/q8" --fit "$fit" --out "$AUF_WORKDIR" --seed "$FIT_SEED"
"$study_python" -u -m experiments.handoff_transfer.train_normal --family q8 --root "$AUF_WORKDIR" --study "$study" --target "$TRANSFER_WORK/models/8b/target" --epochs 3 --lr 1e-3 --seed "$FIT_SEED"
export SPECFORGE_ROOT="$control/experiments/handoff_transfer/specforge"
export AUF_MODELS="$root/models/qwen4" AUF_DOMAIN=math
export PYTHONPATH="$code/port:$code/vendor/code:$control/experiments/handoff_transfer/overlay:$SPECFORGE_ROOT:$snapshot/src:$snapshot"
mkdir -p "$AUF_WORKDIR/logs"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$AUF_WORKDIR/logs/gpu.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
for kind in fusion_r56 five_maps; do
  "$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$AUF_WORKDIR/seed_train.py" --kind "$kind" --tag full --output-tag "gate-$kind-$SLURM_JOB_ID" --epochs 1 --max-steps 2 --seed "$FIT_SEED"
  "$study_python" "$code/verify_transfer.py" --kind "$kind" --tag "gate-$kind-$SLURM_JOB_ID" --steps 2
  "$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$AUF_WORKDIR/seed_train.py" --kind "$kind" --tag full --output-tag "full-$kind" --epochs 4 --max-steps 2000 --seed "$FIT_SEED"
  "$study_python" "$code/verify_transfer.py" --kind "$kind" --tag "full-$kind" --steps 2000
done
