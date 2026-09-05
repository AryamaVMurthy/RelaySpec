#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --nodelist=node01
#SBATCH --job-name=relayspec-setup-deepspec
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail
PROJECT_DIR="${RELAYSPEC_PROJECT_DIR:-$HOME/relayspec}"
: "${RELAYSPEC_PYTHON:?RELAYSPEC_PYTHON must be set}"
: "${DEEPSPEC_PYTHON_OVERLAY:?DEEPSPEC_PYTHON_OVERLAY must be set}"
mkdir -p "$PROJECT_DIR/logs" "$DEEPSPEC_PYTHON_OVERLAY"
"$HOME/.local/bin/uv" pip install \
    --python "$RELAYSPEC_PYTHON" \
    --target "$DEEPSPEC_PYTHON_OVERLAY" \
    --upgrade \
    transformers==5.10.2 \
    numpy==2.4.4 \
    PyYAML==6.0.3 \
    tqdm==4.67.3 \
    typing_extensions==4.15.0 \
    sentencepiece==0.2.1 \
    safetensors==0.7.0 \
    prettytable==3.17.0
PYTHONPATH="$DEEPSPEC_PYTHON_OVERLAY" "$RELAYSPEC_PYTHON" - <<'PY'
import json
import numpy
import transformers

print(json.dumps({
    "numpy": numpy.__version__,
    "transformers": transformers.__version__,
}, sort_keys=True))
PY
