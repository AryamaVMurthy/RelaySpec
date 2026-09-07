#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --gres=gpu:4
#SBATCH --nodelist=node07
#SBATCH --time=00:10:00
#SBATCH --job-name=rs-transfer-probes
#SBATCH --output=/home/aryama.murthy/relayspec-transfer-reproduction/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-transfer-reproduction/logs/%x-%j.err
set -euo pipefail
export TRANSFER_WORK=/scratch/aryama.murthy/transfer-reproduction-20260907/work
cd /scratch/aryama.murthy/transfer-reproduction-20260907/package
.venv/bin/python src/check_environment.py
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
.venv/bin/python /home/aryama.murthy/relayspec-transfer-reproduction/run_probes.py
