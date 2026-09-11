import json
import pytest
from experiments.auf_vllm.select_lr import select


def test_selection_uses_smallest_rate_in_predeclared_band(tmp_path):
    for loss in ['zip','ce','auf']:
        for rate,score in [('1e-4',9.8),('3e-4',9.95),('1e-3',10.)]:
            path=tmp_path/f'q8-n512-{loss}-lr{rate}-s42/offline-validation.json'
            path.parent.mkdir()
            path.write_text(json.dumps({'records':1024,'manifest_sha256':'same',
                'results':[{'epoch':3,'metrics':{'blocks':4096,'accepted_prefix':score}}]}))
    assert all(x['selected_lr']=='3e-4' for x in select(tmp_path).values())
    path=tmp_path/'q8-n512-auf-lr1e-3-s42/offline-validation.json'
    data=json.loads(path.read_text());data['manifest_sha256']='different'
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        select(tmp_path)
