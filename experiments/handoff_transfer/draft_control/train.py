"""Use the unchanged handoff batching, AUF objective, optimizer and schedule."""
from pathlib import Path

source = Path(__file__).resolve().parents[1] / 'port/train.py'
exec(compile(source.read_text(), str(source), 'exec'))
