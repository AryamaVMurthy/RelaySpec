"""Bounded metadata cache and memory-mapped features for full cross fitting."""
import json
from collections import OrderedDict
from pathlib import Path
import torch
from experiments.handoff_transfer.matrix.prepare import digest


class CrossRecords:
    def __init__(self, manifest, cached_chunks=2):
        if cached_chunks < 1:
            raise ValueError('Metadata cache must be positive')
        self.manifest = json.loads(Path(manifest).read_text())
        self.index = self.manifest['index']
        assert len(self.index) == self.manifest['records']
        assert len({r['group_id'] for r in self.index}) == len(self.index)
        self.cached_chunks = cached_chunks
        self.metadata = OrderedDict()
        self.verified = set()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        entry = self.index[item]
        chunk = Path(entry['chunk'])
        if chunk not in self.metadata:
            assert digest(chunk / 'train.json') == entry['rollout_sha256']
            assert digest(chunk / 'source-label-alignment.json') == entry['alignment_sha256']
            self.metadata[chunk] = (json.loads((chunk / 'train.json').read_text()),
                                   json.loads((chunk / 'source-label-alignment.json').read_text()))
            if len(self.metadata) > self.cached_chunks:
                self.metadata.popitem(last=False)
        self.metadata.move_to_end(chunk)
        rows, alignments = self.metadata[chunk]
        row, aligned = rows[entry['row_index']], alignments[entry['row_index']]
        assert row['group_id'] == aligned['group_id'] == entry['group_id']
        feature = Path(entry['feature_path'])
        if feature not in self.verified:
            assert digest(feature) == entry['feature_sha256'], 'Feature cache changed'
            self.verified.add(feature)
        data = torch.load(feature, weights_only=True, mmap=True)
        assert data['group_id'] == entry['group_id'] and data['layers'] == [1, 8, 15, 22, 29]
        assert data['features'].shape == (len(row['full_ids']), 20480)
        return {'target_ids': row['full_ids'], 'features': data['features'], 'alignment': aligned}
