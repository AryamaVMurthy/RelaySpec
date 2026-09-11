#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --job-name=handoff-transfer
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${FAMILY:?Set family}" "${KIND:?Set fusion_r56 or five_maps}"
study_control=/home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer
study_root=/scratch/aryama.murthy/handoff-transfer-20260911
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export SPECFORGE_ROOT="$study_control/specforge"
export PYTHONPATH="$study_control/overlay:$study_control/port:$study_control/vendor/code:$SPECFORGE_ROOT"
export AUF_MODELS="$study_root/models/qwen4" AUF_WORKDIR="$study_root/$FAMILY" AUF_DOMAIN=math
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
training_gpus="$CUDA_VISIBLE_DEVICES"
gate_tag="check-$KIND-$SLURM_JOB_ID"
mkdir -p "$AUF_WORKDIR/features/check" "$AUF_WORKDIR/logs"
ln -sfn ../full/00000.pt "$AUF_WORKDIR/features/check/00000.pt"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5 > "$AUF_WORKDIR/logs/gpu-$KIND-$SLURM_JOB_ID.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid"' EXIT
"$study_python" "$study_control/vendor/code/check_loss.py"
"$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$study_control/port/train.py" --kind "$KIND" --tag check --output-tag "$gate_tag" --epochs 1 --max-steps 2
"$study_python" "$study_control/verify_transfer.py" --kind "$KIND" --tag "$gate_tag" --steps 2
# Validate deployment as well as the training/export tensors before full training.
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
export PYTHONPATH="$control:$PYTHONPATH"
export VLLM_BATCH_INVARIANT=1 HF_HUB_OFFLINE=1
extra=()
if [ "$FAMILY" = q8 ]; then extra+=(--target-path /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target); fi
"$study_python" -m experiments.auf_vllm.prepare_family_benchmark --family "$FAMILY" --models "$study/models" "${extra[@]}" --manifest "$study/manifests/dev.jsonl" --out "$AUF_WORKDIR/evaluation"
# Expose one GPU to inference; the Slurm reservation remains two GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
mode=handoff-r56
if [ "$KIND" = five_maps ]; then mode=handoff-five; fi
for variant in ar "$mode"; do
  export_args=()
  if [ "$variant" != ar ]; then export_args+=(--export "$AUF_WORKDIR/exports/$gate_tag/$KIND"); fi
  "$study_python" -u -m experiments.auf_vllm.family_benchmark --family "$FAMILY" --models "$study/models" "${extra[@]}" "${export_args[@]}" --data "$AUF_WORKDIR/evaluation" --mode "$variant" --count 4 --cap 128 --out "$AUF_WORKDIR/gate-$KIND-$SLURM_JOB_ID"
done
"$study_python" -m experiments.auf_vllm.compare_outputs --ar "$AUF_WORKDIR/gate-$KIND-$SLURM_JOB_ID/ar-r0-w0.jsonl" --method "$AUF_WORKDIR/gate-$KIND-$SLURM_JOB_ID/$mode-r0-w0.jsonl" --out "$AUF_WORKDIR/gate-$KIND-$SLURM_JOB_ID/exactness.json" --require-exact --expected-count 4
export CUDA_VISIBLE_DEVICES="$training_gpus"
if [ "${RUN_FULL:-0}" = 1 ]; then
  "$study_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$study_control/port/train.py" --kind "$KIND" --tag full --output-tag "full-$KIND" --epochs 4 --max-steps 2000
  "$study_python" "$study_control/verify_transfer.py" --kind "$KIND" --tag "full-$KIND" --steps 2000
fi
