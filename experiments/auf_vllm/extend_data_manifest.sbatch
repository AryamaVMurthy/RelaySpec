#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --nodelist=node07
#SBATCH --time=00:10:00
#SBATCH --job-name=auf-data32768-manifest
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
cd /home/aryama.murthy/relayspec-auf-20260911
export PYTHONPATH="$PWD" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
study_original=/scratch/aryama.murthy/transfer-reproduction-20260907/work
/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python -u -m experiments.auf_vllm.extend_data_manifest --original "$study_original/common/dataset" --parquet "$study_original/dataset/train-00000.parquet" --tokenizer "$study_original/models/8b/target" --out /scratch/aryama.murthy/relayspec-auf-20260911/manifests-32768
