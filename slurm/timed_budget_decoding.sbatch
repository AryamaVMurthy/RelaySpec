#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --time=00:10:00
#SBATCH --nodelist=node07
#SBATCH --job-name=rs-budget-decode
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module load u22/cuda/12.4
cd "${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec-campaign-20260905}"
export RELAYSPEC_PYTHON="$HOME/.venvs/factorspec-20260826/bin/python"
export RELAYSPEC_CACHE_DIR="/scratch/$USER/factorspec-runtime-20260826"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export DFLASH_SOURCE="$HOME/factorspec/vendor/dflash"
export DEEPSPEC_SOURCE="$HOME/relayspec/vendor/deepspec"
export DEEPSPEC_PYTHON_OVERLAY="$RELAYSPEC_CACHE_DIR/deepspec-python-overlay-node07"
export RELAYSPEC_OUTPUT="$PWD/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RELAYSPEC_OUTPUT"
export CAMPAIGN_CONFIG="$RELAYSPEC_OUTPUT/campaign-config.yaml"
"$RELAYSPEC_PYTHON" scripts/build_timed_budget_campaign.py --check-files --output "$CAMPAIGN_CONFIG" --runs \
  /home/aryama.murthy/relayspec-timed-budget-1c8a72f/outputs/rs-time-quarter-pilot-27861 \
  /home/aryama.murthy/relayspec-timed-budget-1c8a72f/outputs/rs-time-one-27864 \
  /home/aryama.murthy/relayspec-timed-budget-1c8a72f/outputs/rs-time-four-27865
export CAMPAIGN_CHECK_DUPLICATE=0
family="$("$RELAYSPEC_PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["proposer"]["family"])' "$CAMPAIGN_CONFIG")"
case "$family" in
  dflash) export PYTHONPATH="$PWD/src:$DFLASH_SOURCE" ;;
  eagle3)
    test -d "$DEEPSPEC_PYTHON_OVERLAY/transformers"
    export PYTHONPATH="$PWD/src:$DEEPSPEC_PYTHON_OVERLAY:$DEEPSPEC_SOURCE"
    ;;
  *) echo "Unsupported proposer family: $family" >&2; exit 2 ;;
esac
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export RELAYSPEC_OUTPUT="$PWD/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RELAYSPEC_OUTPUT"
cp SOURCE_COMMIT "$RELAYSPEC_OUTPUT/source-commit.txt"
sha256sum scripts/run_controlled_scaling.py scripts/train_relay.py scripts/benchmark_relay.py src/relayspec/*.py > "$RELAYSPEC_OUTPUT/source-sha256.txt"
tar -czf "$RELAYSPEC_OUTPUT/source-snapshot.tar.gz" src scripts configs slurm SOURCE_COMMIT
"$RELAYSPEC_PYTHON" scripts/check_allocation.py > "$RELAYSPEC_OUTPUT/allocation.jsonl"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used --format=csv -l 10 > "$RELAYSPEC_OUTPUT/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
equality_args=()
if [[ "${CAMPAIGN_CHECK_DUPLICATE:-1}" == 1 ]]; then
  equality_args=(--equal-methods relay_dense_a relay_dense_b)
fi
timeout --signal=TERM --kill-after=10s "${CAMPAIGN_TIMEOUT:-540}s" "$RELAYSPEC_PYTHON" scripts/run_mapper_campaign.py --config "$CAMPAIGN_CONFIG" "${equality_args[@]}"
