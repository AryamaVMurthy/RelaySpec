#!/bin/bash
#SBATCH --account=priyesh.shukla
#SBATCH --partition=u22
#SBATCH --qos=high
#SBATCH --nodelist=node07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --job-name=yashas-access
#SBATCH --output=/home/aryama.murthy/relayspec-auf-20260911/logs/yashas-access-%j.out
#SBATCH --error=/home/aryama.murthy/relayspec-auf-20260911/logs/yashas-access-%j.err
set -euo pipefail
control=/home/aryama.murthy/relayspec-auf-20260911
share=/scratch/aryama.murthy/yashas-handoff-20260912
original=/scratch/aryama.murthy/transfer-reproduction-20260907/package/.venv
mkdir -p "$share/control" "$share/python" "$share/venv/bin"
rsync -a --exclude access-tools "$control/" "$share/control/"
rsync -aL /home/aryama.murthy/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/ "$share/python/"
ln -sfn "$original/lib" "$share/venv/lib"
ln -sfn lib "$share/venv/lib64"
ln -sfn "$share/python/bin/python3.12" "$share/venv/bin/python3.12"
ln -sfn python3.12 "$share/venv/bin/python3"
ln -sfn python3.12 "$share/venv/bin/python"
printf 'home = %s/python/bin\ninclude-system-site-packages = false\nversion = 3.12.13\n' "$share" > "$share/venv/pyvenv.cfg"
"$share/venv/bin/python" -c 'import sys, torch, vllm; print(sys.executable,sys.prefix,sys.base_prefix,torch.__version__,vllm.__version__)'
python3 "$control/access-tools/grant_node07_access.py"
