#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodelist=node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=handoff-reuse-ar
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911
run=/scratch/aryama.murthy/handoff-transfer-20260911/q14
python3 "$control/experiments/handoff_transfer/reuse_ar.py" --source /scratch/aryama.murthy/relayspec-auf-20260911/benchmarks/q14-selected/development128-cap2048/ar-r0-w0.jsonl --gate "$run/gate-fusion_r56/ar-r0-w0.summary.json" --evaluation "$run/evaluation/eval.json" --dest "$run/measurements"
