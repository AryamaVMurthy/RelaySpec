"""Capture the Llama8 target at the audited cross-family taps using vLLM."""
import argparse,json
from pathlib import Path
from experiments.auf_vllm.llama_pilot_data import capture,MODELS
from experiments.auf_vllm.pilot_data import write,sha

def main(out):
    generation=json.loads((out/'train-generation.json').read_text())
    assert generation['teacher']=='unsloth/Llama-3.1-8B-Instruct'
    assert generation['target_adapters'] is None
    assert generation['sha256']==sha(out/'train.json')
    # Existing capture role 'source' selects Llama8 rather than Llama3.
    # In THIS experiment Llama8 is the new target; record that explicitly.
    capture(out,role='source',family='llama')
    write(out/'conditioning-contract.json',{
        'conditioning_model':str(MODELS/'llama8-source'),
        'role_in_this_experiment':'target','storage_role_from_shared_capture':'source',
        'target_taps':[1,8,15,22,29],'feature_width':20480,
        'train_sha256':sha(out/'train.json'),'target_adapters':None,
        'causality_check':json.loads((out/'features/source/train/causality.json').read_text())})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
