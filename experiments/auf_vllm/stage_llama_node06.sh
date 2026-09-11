#!/bin/bash
# Run on the login node: bounded file transfer only, no model execution.
set -euo pipefail
stage_root=/scratch/node06/aryama.murthy/relayspec-auf-20260911
stage_hub=/scratch/node07/aryama.murthy/factorspec-runtime-20260826/huggingface/hub
mkdir -p /scratch/node06/aryama.murthy/transfer-reproduction-20260907/package "$stage_root/models"
rsync -a /scratch/node07/aryama.murthy/transfer-reproduction-20260907/package/.venv/ /scratch/node06/aryama.murthy/transfer-reproduction-20260907/package/.venv/
rsync -aL "$stage_hub/models--unsloth--Llama-3.2-3B-Instruct/snapshots/006f5dcd1393c3add266de40994ba96225e9689d/" "$stage_root/models/llama3-target/"
rsync -aL "$stage_hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5/" "$stage_root/models/llama8-source/"
rsync -aL "$stage_hub/models--z-lab--LLaMA3.1-8B-Instruct-DFlash-UltraChat/snapshots/d3af30def9601abdd10810aba220d692f0e803f0/" "$stage_root/models/llama8-draft/"
echo "Llama assets and pinned runtime staged on node06"
