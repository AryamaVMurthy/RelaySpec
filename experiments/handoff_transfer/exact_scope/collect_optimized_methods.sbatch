#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
campaign="${RELAYSPEC_EVAL_CAMPAIGN:-optimized-all128-20260912}"
out="$control/outputs/$campaign"
mkdir -p "$out"
cd "$control"
export PYTHONPATH="$control/src:$control"
/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python -m experiments.handoff_transfer.exact_scope.collect_optimized_methods --root "$root" --out "$out/results.json" --require-complete
tar --exclude=__pycache__ --exclude=vllm-cache --exclude=inductor-cache -czf "$out/measurements.tar.gz.tmp" -C "$root" "$campaign"
mv "$out/measurements.tar.gz.tmp" "$out/measurements.tar.gz"
sha256sum "$out/measurements.tar.gz" > "$out/measurements.sha256"
