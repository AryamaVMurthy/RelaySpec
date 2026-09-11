#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --job-name=handoff-matrix-array
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
: "${MATRIX_TASKS:?}"
mapfile -t values < <(python3 /home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer/matrix/task.py "$MATRIX_TASKS" "$SLURM_ARRAY_TASK_ID")
[ "${#values[@]}" = 5 ]
case "${values[0]}" in q8|llama) ;; *) echo 'Cross tasks require cross_full_fit.sbatch' >&2; exit 2;; esac
export FAMILY="${values[0]}" KIND="${values[1]}" MATRIX_OBJECTIVE="${values[2]}" MATRIX_LR="${values[3]}" TRAIN_STEPS="${values[4]}"
bash /home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer/matrix_fit.sbatch
