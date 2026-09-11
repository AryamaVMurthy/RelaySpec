#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --nodelist=node07
#SBATCH --time=00:03:00
#SBATCH --job-name=auf-archive-lr
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/%x-%j.out
set -euo pipefail
study_source=/scratch/aryama.murthy/relayspec-auf-20260911/lr-screen
study_dest=/home/aryama.murthy/relayspec-auf-20260911/outputs/lr-screen
for loss in zip ce auf; do
  for rate in 1e-4 3e-4 1e-3; do
    name="q8-n512-${loss}-lr${rate}-s42"
    mkdir -p "$study_dest/$name"
    cp "$study_source/$name/offline-validation.json" "$study_dest/$name/offline-validation.json"
  done
done
