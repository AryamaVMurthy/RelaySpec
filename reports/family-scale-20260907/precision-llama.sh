#!/bin/bash
set -euo pipefail
module load u22/cuda/12.4
cd /home/aryama.murthy/rs-zip-family-20260907
export HF_HOME=/scratch/aryama.murthy/factorspec-runtime-20260826/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export DFLASH_SOURCE="$HOME/factorspec/vendor/dflash"
export PYTHONPATH="$PWD/src:$DFLASH_SOURCE"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export ZIP_FAMILY_OUTPUT=/scratch/aryama.murthy/zip-family-pilot-20260907
PY="$HOME/.venvs/factorspec-20260826/bin/python"
export FAMILY_SCALE_CACHE=/scratch/aryama.murthy/family-scale-20260907
export FAMILY_SCALE_DATA=/scratch/aryama.murthy/factorspec-runtime-20260826/relayspec/scaling-data/numina-v1
export CUDA_VISIBLE_DEVICES=0
for precision in bfloat16 float16; do
  export RELAYSPEC_OUTPUT="$FAMILY_SCALE_CACHE/precision-llama-$precision"
  mkdir -p "$RELAYSPEC_OUTPUT"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=1 scripts/benchmark_mixed_precision_relay.py --config "reports/family-scale-20260907/precision-llama-$precision.yaml" > "$RELAYSPEC_OUTPUT/stdout.log" 2> "$RELAYSPEC_OUTPUT/stderr.log"
done
