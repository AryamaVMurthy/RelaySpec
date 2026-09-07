"""Require complete paired exact-AR comparisons before promoting a run."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = []
for lane in range(4):
    path = root/f'lane{lane}'/'benchmark-summary.json'
    data = json.loads(path.read_text())
    assert data['paired_requests'] > 0
    for name, row in data['methods'].items():
        assert row['requests'] == data['paired_requests']
        assert row['exact_sequence_matches'] == row['requests'], (path, name, row)
        assert row['output_tokens'] > 0 and row['decode_seconds'] > 0
    results.append({'lane':lane, **data})
(root/'correctness-gate.json').write_text(json.dumps({'status':'pass',
    'reference':'matched full-FP32 greedy AR; finite request check',
    'results':results},indent=2))
