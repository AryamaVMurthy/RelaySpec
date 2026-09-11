#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodelist=node07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=handoff-origin-full
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
study_control=/home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer
study_root=/scratch/aryama.murthy/handoff-transfer-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export SPECFORGE_ROOT="$study_control/specforge"
export PYTHONPATH="$study_root/overlay:$study_control/vendor/code:$SPECFORGE_ROOT"
export AUF_MODELS="$study_root/models/qwen4" AUF_WORKDIR="$study_root/origin-math-full" AUF_DOMAIN=math
export AUF_ADAPTER
AUF_ADAPTER=$("$study_python" -c "import json;print(json.load(open('$study_root/prepared.json'))['adapter'])")
export VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=0 AUF_SAFE_RUNTIME=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p "$AUF_WORKDIR/setup"
cp "$study_control/data/math_train.json" "$AUF_WORKDIR/setup/train.json"
cp "$study_control/data/math_eval.json" "$AUF_WORKDIR/setup/eval.json"
"$study_python" "$study_control/vendor/code/check_loss.py"
"$study_python" "$study_control/vendor/code/generate.py" --tag full --count 4096 --cap 4096
"$study_python" "$study_control/vendor/code/capture.py" --tag full
"$study_python" "$study_control/vendor/code/check_inputs.py"
unset AUF_SAFE_RUNTIME
"$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$study_control/vendor/code/train.py" --kind fusion_r56 --tag full --epochs 4 --max-steps 2000
"$study_python" "$study_control/verify_origin_gate.py" --tag full --steps 2000 --records 4096
export AUF_SAFE_RUNTIME=1
for variant in ar native mapper; do
  "$study_python" "$study_control/vendor/code/evaluate.py" --variant "$variant" --count 128 --reference-file "$study_control/data/math_references.json"
done
