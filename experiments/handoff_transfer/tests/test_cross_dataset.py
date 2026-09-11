import json
import pytest
import torch
from experiments.handoff_transfer.cross.dataset import CrossRecords
from experiments.handoff_transfer.matrix.prepare import digest


def test_streamed_features_validate_content_and_metadata(tmp_path):
    chunk = tmp_path / 'chunk'; chunk.mkdir()
    (chunk / 'train.json').write_text(json.dumps([dict(group_id='a', full_ids=[1,2])]))
    (chunk / 'source-label-alignment.json').write_text(json.dumps([dict(group_id='a', blocks=[])]))
    feature = chunk / 'features.pt'
    torch.save(dict(group_id='a', layers=[1,8,15,22,29], features=torch.zeros(2,20480,dtype=torch.bfloat16)), feature)
    entry = dict(group_id='a', chunk=str(chunk), row_index=0, feature_path=str(feature),
        feature_sha256=digest(feature), rollout_sha256=digest(chunk/'train.json'),
        alignment_sha256=digest(chunk/'source-label-alignment.json'))
    manifest = tmp_path/'index.json'
    manifest.write_text(json.dumps(dict(records=1,index=[entry])))
    records = CrossRecords(manifest)
    assert len(records)==1 and records[0]['features'].shape==(2,20480)
    assert len(records.metadata)==len(records.verified)==1
    feature.write_bytes(b'changed')
    with pytest.raises(AssertionError, match='Feature cache changed'):
        CrossRecords(manifest)[0]
