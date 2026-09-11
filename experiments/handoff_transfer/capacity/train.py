"""Run the same training loop with the separate capacity model import."""
from pathlib import Path

source = Path(__file__).resolve().parents[1] / 'port/train.py'
exec(compile(source.read_text(), str(source), 'exec'))
