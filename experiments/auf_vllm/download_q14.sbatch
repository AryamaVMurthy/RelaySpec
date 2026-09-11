#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --nodelist=node06
#SBATCH --time=01:00:00
#SBATCH --job-name=auf-q14-download
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD"
/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python -u -m experiments.auf_vllm.download_q14
