import json
import os
from pathlib import Path
import subprocess
import sys

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def test_parallel_alignment_preserves_serial_label_bytes_and_record_order(tmp_path):
    model_dir=tmp_path/'models'
    for family,words in [('qwen4-source',['f','e','d','c','b','a']),
                         ('llama8-source',['a','b','c','d','e','f'])]:
        backend=Tokenizer(models.WordLevel({'[UNK]':0,**{w:i+1 for i,w in enumerate(words)}},unk_token='[UNK]'))
        backend.normalizer=normalizers.NFC()
        backend.pre_tokenizer=pre_tokenizers.Whitespace()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]')
        tokenizer.save_pretrained(model_dir/family)
    rows=[dict(group_id=f'record-{i}',prompt_token_ids=[1],full_ids=[1,2,3,4,5,6][:6-i%2]) for i in range(5)]
    env=dict(os.environ,TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
    root=Path(__file__).resolve().parents[3]
    outputs=[]
    for workers in (None,2):
        out=tmp_path/str(workers);out.mkdir();(out/'train.json').write_text(json.dumps(rows))
        command=[sys.executable,'-m','experiments.handoff_transfer.cross.prepare','--out',str(out),'--models',str(model_dir)]
        if workers is not None:command+=['--workers',str(workers)]
        result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=60)
        assert result.returncode==0,result.stderr
        outputs.append((out/'source-label-alignment.json').read_bytes())
    assert outputs[0]==outputs[1]
    assert [r['group_id'] for r in json.loads(outputs[1])]==[r['group_id'] for r in rows]
