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
#SBATCH --job-name=rs-full-cache
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module load u22/cuda/12.4
cd "${RELAYSPEC_PROJECT_DIR:?explicit immutable source snapshot required}"
export RELAYSPEC_PYTHON="$HOME/.venvs/factorspec-20260826/bin/python"
export RELAYSPEC_CACHE_DIR="/scratch/$USER/factorspec-runtime-20260826"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export DFLASH_SOURCE="$HOME/factorspec/vendor/dflash"
export DEEPSPEC_SOURCE="$HOME/relayspec/vendor/deepspec"
export DEEPSPEC_PYTHON_OVERLAY="$RELAYSPEC_CACHE_DIR/deepspec-python-overlay-node07"
family="$("$RELAYSPEC_PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["proposer"]["family"])' "${CACHE_TRAINING_CONFIG:-configs/protocol_active/train_dflash_qwen3_8b_relative_4gpu.yaml}")"
case "$family" in
  dflash) export PYTHONPATH="$PWD/src:$DFLASH_SOURCE" ;;
  eagle3)
    test -d "$DEEPSPEC_PYTHON_OVERLAY/transformers"
    export PYTHONPATH="$PWD/src:$DEEPSPEC_PYTHON_OVERLAY:$DEEPSPEC_SOURCE"
    ;;
  *) echo "Unsupported proposer family: $family" >&2; exit 2 ;;
esac
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export RELAYSPEC_OUTPUT="$PWD/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
export RELAYSPEC_FEATURE_CACHE="$RELAYSPEC_CACHE_DIR/relayspec/feature-caches/${RELAYSPEC_CACHE_TAG:?explicit cache tag required}-${SLURM_JOB_ID}"
: "${RELAYSPEC_PILOT_GATE:?explicit successful family pilot required}"
extraction_args=()
if [[ -n "${CACHE_TRAINING_CONFIG:-}" ]]; then
  extraction_args+=(--training-config "$CACHE_TRAINING_CONFIG")
fi
if [[ -n "${CACHE_DATA_ROOT:-}" ]]; then
  extraction_args+=(--data-root "$CACHE_DATA_ROOT")
fi
if [[ -n "${CACHE_TRAINING_MANIFEST:-}" ]]; then
  extraction_args+=(--training-manifest "$CACHE_TRAINING_MANIFEST")
fi
if [[ "${CACHE_REPORT_DOMAINS:-0}" == 1 ]]; then
  extraction_args+=(--report-domain-diagnostics)
fi
mkdir -p "$RELAYSPEC_OUTPUT"
cp SOURCE_COMMIT "$RELAYSPEC_OUTPUT/source-commit.txt"
tar -czf "$RELAYSPEC_OUTPUT/source-snapshot.tar.gz" src scripts configs slurm SOURCE_COMMIT
"$RELAYSPEC_PYTHON" scripts/check_allocation.py > "$RELAYSPEC_OUTPUT/allocation.jsonl"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used --format=csv -l 10 > "$RELAYSPEC_OUTPUT/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
timeout --signal=TERM --kill-after=10s 540s "$RELAYSPEC_PYTHON" scripts/extract_scaling_cache.py \
  --pilot-gate "$RELAYSPEC_PILOT_GATE" --train-records "${RELAYSPEC_TRAIN_RECORDS:-2048}" \
  --validation-records "${RELAYSPEC_VALIDATION_RECORDS:-1024}" "${extraction_args[@]}"
