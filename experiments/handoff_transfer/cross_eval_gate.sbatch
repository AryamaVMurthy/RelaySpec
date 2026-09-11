#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=handoff-cross-decode
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
run=/scratch/aryama.murthy/handoff-transfer-20260911/anchors512/cross/pilot16
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export PYTHONPATH="$control" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
export CROSS_SOURCE_TOKENIZER="$study/models/qwen4-source"
target="$study/models/llama8-source"
# Fail before loading AR when either training/export prerequisite is absent.
for objective in ce auf; do
 fit="$run/cross-fits/five_maps-$objective"
 "$study_python" -c "import json;assert json.load(open('$fit/verification.json'))['status']=='training_and_export_verified'"
 test -s "$fit/export/model.safetensors"
done
"$study_python" -m experiments.auf_vllm.prepare_family_benchmark --family cross --models "$study/models" --target-path "$target" --manifest "$study/manifests/dev.jsonl" --out "$run/evaluation"
"$study_python" -u -m experiments.auf_vllm.family_benchmark --family cross --target-path "$target" --models "$study/models" --data "$run/evaluation" --mode ar --count 4 --cap 128 --out "$run/decode-gate/ar"
for objective in ce auf; do
 fit="$run/cross-fits/five_maps-$objective"
 "$study_python" -c "import json;assert json.load(open('$fit/verification.json'))['status']=='training_and_export_verified'"
 "$study_python" -u -m experiments.auf_vllm.family_benchmark --family cross --target-path "$target" --models "$study/models" --data "$run/evaluation" --export "$fit/export" --mode matrix --count 4 --cap 128 --out "$run/decode-gate/$objective"
 "$study_python" -m experiments.auf_vllm.compare_outputs --ar "$run/decode-gate/ar/ar-r0-w0.jsonl" --method "$run/decode-gate/$objective/matrix-r0-w0.jsonl" --out "$run/decode-gate/$objective/exactness.json" --expected-count 4 --require-exact
done
