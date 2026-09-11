#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=node07
#SBATCH --time=05:00:00
#SBATCH --job-name=handoff-seed-eval
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.err
set -euo pipefail
: "${FIT_SEED:?Set43 or44}"
[[ "$FIT_SEED" = 43 || "$FIT_SEED" = 44 ]]
control=/home/aryama.murthy/relayspec-auf-20260911
study=/scratch/aryama.murthy/relayspec-auf-20260911
root=/scratch/aryama.murthy/handoff-transfer-20260911
run="$root/seeds/q8-s$FIT_SEED"
study_python=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv/bin/python
export PYTHONPATH="$control"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1
for kind in fusion_r56 five_maps; do
 "$study_python" -c "import json;assert json.load(open('$run/modules/full-$kind/$kind/verification.json'))['status']=='passed';assert json.load(open('$run/modules/full-$kind/$kind/summary.json'))['seed']==int('$FIT_SEED')"
done
for repeat in 0 1 2; do
 modes=(normal zip handoff-r56 handoff-five)
 for index in 0 1 2 3; do
  mode=${modes[$(( (index + repeat) % 4 ))]}
  case "$mode" in
   normal) model="$run/normal/export";;
   zip) model="$root/seed-initializers/q8-s$FIT_SEED/epoch-3/export";;
   handoff-r56) model="$run/exports/full-fusion_r56/fusion_r56";;
   handoff-five) model="$run/exports/full-five_maps/five_maps";;
  esac
  "$study_python" -u -m experiments.auf_vllm.family_benchmark --family q8 --models "$study/models" --target-path /scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target --export "$model" --data "$root/q8/evaluation" --mode "$mode" --count 128 --cap 2048 --repeat "$repeat" --out "$run/measurements"
  "$study_python" -m experiments.auf_vllm.compare_outputs --ar "$root/q8/measurements/ar-r$repeat-w0.jsonl" --method "$run/measurements/$mode-r$repeat-w0.jsonl" --out "$run/comparisons/$mode-r$repeat.json" --require-exact --expected-count 128
 done
done
