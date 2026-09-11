import json
import pytest
from experiments.handoff_transfer.matrix.batch_final import frozen_exports
from experiments.handoff_transfer.matrix.prepare import digest


def test_final_batch_rejects_replaced_control(tmp_path):
    base=tmp_path/'q8';base.mkdir()
    controls={}
    for kind in ('normal','zip'):
        folder=base/kind/'export';folder.mkdir(parents=True)
        (folder/'model.safetensors').write_bytes(kind.encode());controls[kind]=digest(folder/'model.safetensors')
    (base/'transfer.json').write_text(json.dumps({'base_export':str(base/'zip/export')}))
    for loss in ('ce','auf'):
        run=tmp_path/'matrix/q8'/f'five_ba56-{loss}-lr0.0001'
        export=run/'exports/steps-2000/five_ba56';export.mkdir(parents=True)
        (export/'model.safetensors').write_bytes(loss.encode())
        result=dict(status='complete_verified',cell=dict(family='q8',kind='five_ba56',objective=loss),
            verification={'export_sha256':digest(export/'model.safetensors')},control_export_sha256=controls)
        (run/'final-comparison.json').write_text(json.dumps(result))
    assert set(frozen_exports(tmp_path))=={'ar','normal','zip','ce','auf'}
    (base/'normal/export/model.safetensors').write_bytes(b'changed')
    with pytest.raises(AssertionError,match='Control differs'):frozen_exports(tmp_path)
