#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --job-name=cross-eval-array
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
: "${MATRIX_TASKS:?}" "${EVAL_SCOPE:?}"
code=/home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer
mapfile -t values < <(python3 "$code/matrix/task.py" "$MATRIX_TASKS" "$SLURM_ARRAY_TASK_ID")
[ "${#values[@]}" = 5 ] && [ "${values[0]}" = cross ]
export MODE=matrix KIND="${values[1]}" MATRIX_OBJECTIVE="${values[2]}" MATRIX_LR="${values[3]}" TRAIN_STEPS="${values[4]}"
bash "$code/cross_benchmark.sbatch"
