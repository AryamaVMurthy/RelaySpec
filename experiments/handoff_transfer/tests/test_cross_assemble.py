import json
import pytest
from experiments.handoff_transfer.cross.assemble import assemble
from experiments.handoff_transfer.matrix.prepare import digest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def chunk(root, offset, key):
    root.mkdir()
    put(root/'train.json', [dict(group_id=key, prompt_token_ids=[1], output_ids=[2], full_ids=[1,2])])
    rollout=digest(root/'train.json')
    put(root/'train-generation.json', dict(teacher='unsloth/Llama-3.1-8B-Instruct', target_adapters=None,
        output_cap=4096, temperature=0, seed=42, offset=offset, records=1, sha256=rollout, source_manifest_sha256='shared'))
    put(root/'source-label-alignment.json', [dict(group_id=key, blocks=[dict(target_anchor=1)])])
    put(root/'alignment-summary.json', dict(rollout_sha256=rollout, labels_sha256=digest(root/'source-label-alignment.json')))
    put(root/'conditioning-contract.json', dict(train_sha256=rollout, causality_check=dict(passed=True),target_taps=[1,8,15,22,29]))
    feature=root/'features/source/train/00000.pt';feature.parent.mkdir(parents=True);feature.write_bytes(b'opaque feature file')
    put(feature.with_suffix('.json'), dict(manifest_sha256=rollout,sha256=digest(feature)))
    return root


def test_complete_index_and_reject_duplicate_or_noncontiguous_chunks(tmp_path):
    a=chunk(tmp_path/'a',0,'a');b=chunk(tmp_path/'b',1,'b')
    result=assemble([a,b],records=2)
    assert result['records']==2 and [r['group_id'] for r in result['index']]==['a','b']
    with pytest.raises(AssertionError,match='Noncontiguous'):assemble([b,a],records=2)
    duplicate=chunk(tmp_path/'duplicate',1,'a')
    with pytest.raises(AssertionError):assemble([a,duplicate],records=2)
    with pytest.raises(AssertionError):assemble([a],records=2)


def test_changed_rollout_refuses_index(tmp_path):
    a=chunk(tmp_path/'a',0,'a')
    put(a/'train.json',[dict(group_id='a',prompt_token_ids=[1],output_ids=[9],full_ids=[1,9])])
    with pytest.raises(AssertionError):assemble([a],records=1)
