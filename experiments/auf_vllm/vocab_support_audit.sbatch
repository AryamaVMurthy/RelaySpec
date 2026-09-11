#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --nodelist=node06
#SBATCH --time=00:15:00
#SBATCH --job-name=auf-cross-vocab-audit
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
export PYTHONPATH="$PWD" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python -u -m experiments.auf_vllm.vocab_support_audit --models /scratch/aryama.murthy/relayspec-auf-20260911/models --bridge assets/qwen3-4b_llama3.1-8b.json --out "outputs/cross-vocab-support-${SLURM_JOB_ID}.json"
