#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${MODE:?}"
control=/home/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
snapshot="$root/transformers-e1b8/source-$SLURM_JOB_ID"
mkdir -p "$snapshot/experiments" "$snapshot/src"
rsync -a --exclude __pycache__ --exclude reports --exclude specforge --exclude overlay --exclude vendor "$control/experiments/handoff_transfer" "$control/experiments/auf_vllm" "$snapshot/experiments/"
rsync -a --exclude __pycache__ "$control/src/relayspec" "$snapshot/src/"
export PYTHONPATH="$snapshot/src:$snapshot" OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
extra=()
if [ "$MODE" != ar ]; then extra+=(--export "$root/matrix32e1b8/q8/five_maps-$MODE-lr0.0001/exports/steps-512/five_maps"); fi
out="$root/transformers-e1b8/q8/$MODE"
mkdir -p "$out"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$out/gpu-$SLURM_JOB_ID.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
cd "$snapshot"
"$python" -u -m experiments.handoff_transfer.transformers_benchmark --family q8 --target /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target --data "$root/exact32e1b8-results/q8/evaluation" --out "$out" --dflash-source /home/aryama.murthy/factorspec/vendor/dflash --mode "$MODE" "${extra[@]}" --count 128 --cap 2048
