#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=handoff-normal
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${FAMILY:?Set family}"
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
run=/scratch/aryama.murthy/handoff-transfer-20260911/$FAMILY
snapshot="$run/source-normal-$SLURM_JOB_ID"
mkdir -p "$snapshot/experiments" "$snapshot/src" "$run/logs"
rsync -a --exclude reports --exclude __pycache__ --exclude overlay --exclude specforge --exclude vendor --exclude data "$control/experiments/auf_vllm" "$control/experiments/handoff_transfer" "$snapshot/experiments/"
rsync -a --exclude __pycache__ "$control/src/relayspec" "$snapshot/src/"
export PYTHONPATH="$snapshot/src:$snapshot"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export DFLASH_SOURCE=/home/aryama.murthy/factorspec/vendor/dflash
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
case "$FAMILY" in
 q8) target="$TRANSFER_WORK/models/8b/target";;
 q14) target="$study/models/qwen14-target";;
 llama) target="$study/models/llama3-target";;
 *) exit 2;;
esac
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$run/logs/normal-gpu-$SLURM_JOB_ID.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
"$study_python" -u -m experiments.handoff_transfer.train_normal --family "$FAMILY" --root "$run" --study "$study" --target "$target" --epochs 3 --lr 1e-3 --seed 42
"$study_python" -m experiments.auf_vllm.prepare_family_benchmark --family "$FAMILY" --models "$study/models" --target-path "$target" --manifest "$study/manifests/dev.jsonl" --out "$run/evaluation"
for mode in ar normal; do
 extra=()
 if [ "$mode" = normal ]; then extra+=(--export "$run/normal/export"); fi
 "$study_python" -u -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --models "$study/models" --target-path "$target" "${extra[@]}" --data "$run/evaluation" --mode "$mode" --count 4 --cap 128 --out "$run/gate-normal"
done
"$study_python" -m experiments.auf_vllm.compare_outputs --ar "$run/gate-normal/ar-r0-w0.jsonl" --method "$run/gate-normal/normal-r0-w0.jsonl" --out "$run/gate-normal/exactness.json" --require-exact --expected-count 4
for repeat in 0 1 2; do
 "$study_python" -u -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --models "$study/models" --target-path "$target" --export "$run/normal/export" --data "$run/evaluation" --mode normal --count 128 --cap 2048 --repeat "$repeat" --out "$run/measurements"
done
