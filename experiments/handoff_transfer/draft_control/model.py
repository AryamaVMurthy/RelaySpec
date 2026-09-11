"""Separate drafter-body LoRA control; immutable AUF loss and frozen ZIP fc."""
import os
from pathlib import Path
from experiments.handoff_transfer.draft_control_source import draft_variant

control_source = Path(__file__).resolve().parents[1] / 'port/model.py'
exec(compile(draft_variant(control_source.read_text(), int(os.environ.get('AUF_DRAFT_RANK', '32'))), str(control_source), 'exec'))
