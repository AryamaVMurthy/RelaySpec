#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodelist=node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=handoff-vocab
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
study=/scratch/aryama.murthy/relayspec-auf-20260911
control=/home/aryama.murthy/relayspec-auf-20260911
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python "$control/experiments/handoff_transfer/audit_cross_vocab.py" --source "$study/models/qwen4-source" --target "$study/models/llama8-source" --manifest "$study/manifests/dev.jsonl" --out "$control/outputs/handoff-transfer/cross-vocabulary-audit.json"
