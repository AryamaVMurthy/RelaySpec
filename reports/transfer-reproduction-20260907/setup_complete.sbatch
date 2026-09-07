#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --nodelist=node07
#SBATCH --time=02:00:00
#SBATCH --job-name=rs-transfer-install
#SBATCH --output=/home/aryama.murthy/relayspec-transfer-reproduction/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-transfer-reproduction/logs/%x-%j.err
set -euo pipefail
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export TMPDIR=/scratch/aryama.murthy/transfer-reproduction-20260907/tmp
cd /scratch/aryama.murthy/transfer-reproduction-20260907/package
.venv/bin/python /home/aryama.murthy/relayspec-transfer-reproduction/resume_install.py
