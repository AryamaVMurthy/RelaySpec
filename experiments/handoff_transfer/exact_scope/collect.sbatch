#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911
export PYTHONPATH="$control"
python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
"$python" -m experiments.handoff_transfer.exact_scope.collect --root /scratch/aryama.murthy/handoff-transfer-20260911 --out "$control/outputs/exact32e1b8/final-audit.json" --require-complete
