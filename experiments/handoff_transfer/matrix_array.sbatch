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
mapfile -t values < <(python3 - "$MATRIX_TASKS" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))[int(sys.argv[2])]
assert r['family'] in ['q8','llama'] and r['objective'] in ['ce','auf']
for key in ['family','kind','objective']:print(r[key])
PY
)
[ "${#values[@]}" = 3 ]
export FAMILY="${values[0]}" KIND="${values[1]}" MATRIX_OBJECTIVE="${values[2]}" MATRIX_LR=0.0001 TRAIN_STEPS=100
bash /home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer/matrix_fit.sbatch
