#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --nodelist=node07
#SBATCH --time=00:30:00
#SBATCH --job-name=auf-q8-prepare16k
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
study_snapshot="outputs/source-${SLURM_JOB_ID}"
mkdir -p "$study_snapshot/experiments"
cp -r experiments/auf_vllm "$study_snapshot/experiments/"
export PYTHONPATH="$PWD/$study_snapshot" OMP_NUM_THREADS=2
study_root=/scratch/aryama.murthy/relayspec-auf-20260911
/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python -u -m experiments.auf_vllm.prepare_q8_large_cache --work /scratch/aryama.murthy/transfer-reproduction-20260907/work --small "$study_root/main-q8-n4096" --out "$study_root/main-q8-n16384"
