#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node06
#SBATCH --time=00:45:00
#SBATCH --job-name=auf-q14-recover
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VLLM_BATCH_INVARIANT=1 HF_HUB_OFFLINE=1
study_snapshot="outputs/source-${SLURM_JOB_ID}"
mkdir -p "$study_snapshot/experiments"
cp -r experiments/auf_vllm "$study_snapshot/experiments/"
export PYTHONPATH="$PWD/$study_snapshot/experiments/auf_vllm/bootstrap:$PWD/$study_snapshot"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
study_out=/scratch/aryama.murthy/relayspec-auf-20260911/main-q14-n4096/part-1
cd "$study_snapshot"
"$study_python" -u -m experiments.auf_vllm.verify_family_capture --out "$study_out" --role target
AUF_CAPTURE=1 "$study_python" -u -m experiments.auf_vllm.llama_pilot_data source --family q14 --out "$study_out"
"$study_python" -u -m experiments.auf_vllm.verify_family_capture --out "$study_out" --role source
