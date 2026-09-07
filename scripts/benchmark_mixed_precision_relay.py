"""Explicit speed diagnostic: FP32 target/mapper, BF16 frozen drafter interface.

Wrap the existing benchmark without changing its verifier or AR implementation.
Only valid for the source-unloaded, two-model relay configs used by this pilot.
"""
import json
import os
import runpy
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
import relayspec.dflash as dflash
from relayspec.relay import TargetFeatureRelay


def main():
    loads=[]
    original_load=AutoModelForCausalLM.from_pretrained
    def load(*args,**kwargs):
        index=len(loads)
        if index not in (0,1):raise RuntimeError('Mixed pilot expects exactly source then target')
        dtype=torch.bfloat16 if index==0 else torch.float32
        kwargs['dtype']=dtype
        m=original_load(*args,**kwargs)
        assert m.get_input_embeddings().weight.dtype==dtype
        assert m.get_output_embeddings().weight.dtype==dtype
        loads.append({'role':'source' if index==0 else 'target','model':str(args[0]),'dtype':str(dtype)})
        return m
    AutoModelForCausalLM.from_pretrained=load
    original_import=dflash.import_official_dflash
    def import_draft(*args,**kwargs):
        cls,generate=original_import(*args,**kwargs)
        load_draft=cls.from_pretrained
        def draft_load(*a,**kw):
            kw['dtype']=torch.bfloat16
            m=load_draft(*a,**kw)
            assert next(m.parameters()).dtype==torch.bfloat16
            return m
        cls.from_pretrained=draft_load
        return cls,generate
    dflash.import_official_dflash=import_draft
    original_forward=TargetFeatureRelay.forward
    def forward(self,x):
        assert x.dtype==torch.float32
        return original_forward(self,x).to(torch.bfloat16)
    TargetFeatureRelay.forward=forward
    runpy.run_path(str(Path(__file__).with_name('benchmark_relay.py')),run_name='__main__')
    assert len(loads)==2
    (Path(os.environ['RELAYSPEC_OUTPUT'])/'mixed-precision.json').write_text(json.dumps({'loads':loads,'target':'float32','mapper':'float32','source_embedding_and_head':'bfloat16','drafter':'bfloat16','mapper_output_cast':'bfloat16','verifier_changed':False},indent=2))

if __name__=='__main__':main()
