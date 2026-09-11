#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --job-name=handoff-eval-pool
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%A_%a.err
set -euo pipefail
: "${POOL:?Set q8 or families}"
if [ "$POOL" = q8 ]; then
 families=(q8 q8 q8 q8 q8 q8)
 modes=(ar zip handoff-r56 handoff-five native handoff-ce)
else
 test "$POOL" = families
 families=(q14 llama q14 llama q14 llama q14 llama)
 modes=(ar ar zip zip handoff-r56 handoff-r56 handoff-five handoff-five)
fi
export FAMILY=${families[$SLURM_ARRAY_TASK_ID]} MODE=${modes[$SLURM_ARRAY_TASK_ID]}
bash /home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer/benchmark.sbatch
