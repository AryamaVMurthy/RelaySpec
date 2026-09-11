#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=handoff-matrix-screen
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${FAMILY:?}" "${KIND:?}" "${MATRIX_OBJECTIVE:?}" "${MATRIX_LR:?}" "${SCREEN_STEPS:?}"
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
run="$root/matrix/$FAMILY/$KIND-$MATRIX_OBJECTIVE-lr$MATRIX_LR"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export PYTHONPATH="$control" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
case "$FAMILY" in
 q8) target=/scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target;;
 llama) target="$study/models/llama3-target";;
 *) exit 2;;
esac
# Job-private prompt files avoid parallel writers racing on shared temp names.
data="$run/validation-data-$SLURM_JOB_ID"
"$study_python" -m experiments.handoff_transfer.matrix.validation --family "$FAMILY" --target "$target" --source "$study/manifests/eval.jsonl" --train "$study/manifests/train.jsonl" --final "$study/manifests/dev.jsonl" --out "$data"
"$study_python" -c "import json;v=json.load(open('$run/modules/steps-$SCREEN_STEPS/$KIND/verification.json'));assert v['status']=='passed'"
for mode in ar matrix; do
 extra=()
 if [ "$mode" = matrix ]; then extra+=(--export "$run/exports/steps-$SCREEN_STEPS/$KIND"); fi
 "$study_python" -u -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --target-path "$target" --models "$study/models" "${extra[@]}" --data "$data" --mode "$mode" --count 32 --cap 512 --out "$run/screen-$SCREEN_STEPS"
done
"$study_python" -m experiments.auf_vllm.compare_outputs --ar "$run/screen-$SCREEN_STEPS/ar-r0-w0.jsonl" --method "$run/screen-$SCREEN_STEPS/matrix-r0-w0.jsonl" --out "$run/screen-$SCREEN_STEPS/comparison.json" --expected-count 32 --require-exact
