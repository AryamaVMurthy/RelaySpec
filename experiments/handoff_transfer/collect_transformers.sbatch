#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=handoff-transformers-collect
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${FAMILY:?Set family}" "${TRANSFORMERS_JOB:?Set source job}"
control=/home/aryama.murthy/relayspec-auf-20260911
run=/scratch/aryama.murthy/handoff-transfer-20260911/$FAMILY
export PYTHONPATH="$control"
python3 -m experiments.handoff_transfer.collect_transformers --family "$FAMILY" --root "$run/transformers-$TRANSFORMERS_JOB" --primary "$run/measurements" --out "$control/outputs/handoff-transfer/$FAMILY-transformers-comparison.json"
