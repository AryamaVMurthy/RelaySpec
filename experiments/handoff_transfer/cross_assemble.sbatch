#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=cross-assemble
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911/cross-full4096
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export PYTHONPATH="$control" OMP_NUM_THREADS=2
"$study_python" -m experiments.handoff_transfer.cross.assemble --chunks "$root"/chunks/????? --records 4096 --out "$root/index.json"
