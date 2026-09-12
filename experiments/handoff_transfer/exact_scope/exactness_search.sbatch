#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
profiles=(invariant-o3-rms invariant-o3-all invariant-smalltile-rms invariant-smalltile-all)
export RELAYSPEC_NUMERICS="${profiles[$SLURM_ARRAY_TASK_ID]}"
out="$root/exactness-search-20260912/$RELAYSPEC_NUMERICS"
snapshot="$out/source-$SLURM_JOB_ID"
mkdir -p "$snapshot/experiments" "$snapshot/src"
rsync -a --exclude reports --exclude __pycache__ --exclude overlay --exclude specforge --exclude vendor --exclude data "$control/experiments/auf_vllm" "$control/experiments/handoff_transfer" "$snapshot/experiments/"
rsync -a --exclude __pycache__ "$control/src/relayspec" "$snapshot/src/"
export PYTHONPATH="$snapshot/experiments/auf_vllm/runtime_zip:$snapshot/src:$snapshot" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_BATCH_INVARIANT=0
if [[ "$RELAYSPEC_NUMERICS" == invariant-* ]]; then export VLLM_BATCH_INVARIANT=1; fi
export VLLM_CACHE_ROOT="$out/vllm-cache" TORCHINDUCTOR_CACHE_DIR="$out/inductor-cache" TORCHINDUCTOR_COMPILE_THREADS=4
cd "$snapshot"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$out/gpu.csv" &
monitor=$!
trap 'kill "$monitor"' EXIT
"$python" -u -m experiments.handoff_transfer.exact_scope.exactness_search
