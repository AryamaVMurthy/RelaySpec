#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --job-name=matrix-baselines
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
: "${FAMILY:?}"
case "$FAMILY" in q8|llama) ;; *) exit 2;; esac
modes=(ar zip)
if [ "$FAMILY" = q8 ]; then modes+=(native); fi
export MODE="${modes[$SLURM_ARRAY_TASK_ID]}"
# Existing completed repetitions are retained by the benchmark's strict contract
# check. These tasks fill missing AR/ZIP controls without retraining a baseline.
bash /home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer/benchmark.sbatch
