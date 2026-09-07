#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodelist=node07
#SBATCH --time=00:30:00
#SBATCH --job-name=rs-transfer-setup
#SBATCH --output=/home/aryama.murthy/relayspec-transfer-reproduction/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-transfer-reproduction/logs/%x-%j.err
set -euo pipefail
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export TMPDIR=/scratch/aryama.murthy/transfer-reproduction-20260907/tmp
mkdir -p "$TMPDIR" "$TRANSFER_WORK"
cd /scratch/aryama.murthy/transfer-reproduction-20260907/package
/home/aryama.murthy/.venvs/factorspec-20260826/bin/python verify_bundle.py
/home/aryama.murthy/.venvs/factorspec-20260826/bin/python bootstrap.py
.venv/bin/python -B -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python src/check_environment.py > "$TRANSFER_WORK/environment-check.json"
