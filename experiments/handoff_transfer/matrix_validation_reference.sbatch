#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=matrix-validation-ar
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${FAMILY:?}"
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export PYTHONPATH="$control" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
case "$FAMILY" in
 q8) target=/scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target;;
 llama) target="$study/models/llama3-target";;
 cross) target="$study/models/llama8-source";;
 *) exit 2;;
esac
out="$root/matrix/$FAMILY/validation-reference"
"$study_python" -m experiments.handoff_transfer.matrix.validation --family "$FAMILY" --target "$target" --source "$study/manifests/eval.jsonl" --train "$study/manifests/train.jsonl" --final "$study/manifests/dev.jsonl" --out "$out/data"
"$study_python" -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --target-path "$target" --models "$study/models" --data "$out/data" --mode ar --count 32 --cap 512 --out "$out"
