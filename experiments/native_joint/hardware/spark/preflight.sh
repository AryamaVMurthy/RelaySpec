#!/usr/bin/env bash
# Read-only host inspection. Does not change drivers, running jobs or power limits.
set -uo pipefail
hostname
uname -a
nvidia-smi -q
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
free -h
df -h /home /tmp
python3 --version
command -v uv
command -v docker
docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}'
