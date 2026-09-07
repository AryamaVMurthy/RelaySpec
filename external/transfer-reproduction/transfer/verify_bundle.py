"""Validate the source-only distribution before installing dependencies."""
from pathlib import Path
import hashlib
R=Path(__file__).resolve().parent
for line in (R/'MANIFEST.sha256').read_text().splitlines():
 expected,name=line.split('  ',1);p=R/name
 assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
print('All distributed files match MANIFEST.sha256')
