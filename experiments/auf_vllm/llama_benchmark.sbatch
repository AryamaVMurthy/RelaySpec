#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node06
#SBATCH --time=00:10:00
#SBATCH --job-name=auf-llama-pilot-decode
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false VLLM_BATCH_INVARIANT=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/experiments/auf_vllm/runtime_zip:$PWD"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
study_root=/scratch/aryama.murthy/relayspec-auf-20260911
"$study_python" -u -m experiments.auf_vllm.llama_benchmark --mode ar --out "outputs/llama-diagnostic-${SLURM_JOB_ID}-ar.json"
for loss in auf ce zip; do
  "$study_python" -u -m experiments.auf_vllm.llama_benchmark --mode "$loss" --export "$study_root/pilot-l3-fit-31275/$loss/export" --out "outputs/llama-diagnostic-${SLURM_JOB_ID}-${loss}.json"
done
