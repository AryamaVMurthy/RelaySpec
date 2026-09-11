#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodelist=node07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=handoff-prepare
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
study_control=/home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer
study_root=/scratch/aryama.murthy/handoff-transfer-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
mkdir -p "$study_root/overlay"
# Isolated additional modules; the live shared runtime is not mutated.
/home/aryama.murthy/.local/bin/uv pip install --python "$study_python" --target "$study_root/overlay" --no-deps peft==0.20.0 accelerate==1.12.0
export SPECFORGE_ROOT="$study_control/specforge"
export PYTHONPATH="$study_root/overlay:$study_control/vendor/code:$SPECFORGE_ROOT"
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
"$study_python" "$study_control/prepare_origin.py"
