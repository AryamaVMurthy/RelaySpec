"""Rank ablation; the pinned rank56 transfer implementation remains untouched."""
import os
from pathlib import Path
from experiments.handoff_transfer.capacity_source import rank_variant

capacity_rank = int(os.environ['AUF_CAPACITY_RANK'])
capacity_source = Path(__file__).resolve().parents[1] / 'port/model.py'
exec(compile(rank_variant(capacity_source.read_text(), capacity_rank), str(capacity_source), 'exec'))
