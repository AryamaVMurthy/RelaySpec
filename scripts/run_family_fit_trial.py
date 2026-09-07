import json
from pathlib import Path
import subprocess
import sys

subprocess.run([sys.executable,*json.loads(Path(sys.argv[1]).read_text())],check=True)
