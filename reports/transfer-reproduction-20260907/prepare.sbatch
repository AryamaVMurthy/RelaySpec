#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodelist=node07
#SBATCH --time=00:20:00
#SBATCH --job-name=rs-transfer-prepare
#SBATCH --output=/home/aryama.murthy/relayspec-transfer-reproduction/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-transfer-reproduction/logs/%x-%j.err
set -euo pipefail
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
export HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=4
cd /scratch/aryama.murthy/transfer-reproduction-20260907/package
.venv/bin/python /home/aryama.murthy/relayspec-transfer-reproduction/materialize_inputs.py
.venv/bin/python reproduce.py prepare
