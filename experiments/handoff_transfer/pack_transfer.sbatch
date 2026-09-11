#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=handoff-pack
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${FAMILY:?Set q8, q14 or llama}"
study_control=/home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export OMP_NUM_THREADS=2
"$study_python" "$study_control/pack_transfer.py" --family "$FAMILY" --study /scratch/aryama.murthy/relayspec-auf-20260911 --out "/scratch/aryama.murthy/handoff-transfer-20260911/$FAMILY"
