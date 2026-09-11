"""Build a clearly labeled cross pilot initializer from preserved ZIP maps.

The historical 512-record initializer is for an integration pilot only. It is
not represented as a new 4,096-record rollout fit.
"""
import argparse,json
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from experiments.handoff_transfer.matrix.prepare import digest
from experiments.auf_vllm.pilot_data import write

def main(a):
    metadata=json.loads((a.archived/'summary.json').read_text())
    assert metadata['pair']=='cross' and metadata['records']==512
    checkpoint=a.archived/'maps.pt'
    assert digest(checkpoint)=='0b3054d269790b0466e162a5df189d2224688f6a9074230a196a251cee2d1666', 'Historical initializer differs from preservation ledger'
    maps=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)['model']
    # The source head, embedding and backbone come from the audited Qwen pair.
    reference=json.loads((a.reference/'transfer.json').read_text())
    state=load_file(str(Path(reference['base_export'])/'model.safetensors'))
    assert torch.equal(maps['norm'],state['hidden_norm.weight'])
    fusion=maps['fusion'];width=2560
    native=load_file(str(Path(reference['draft'])/'model.safetensors'))
    assert torch.equal(fusion,native['fc.weight'])
    folded=torch.cat([fusion[:,i*width:(i+1)*width].float()@maps[f'maps.{i}.weight'].float() for i in range(5)],dim=1)
    assert folded.shape==(2560,20480)
    state['fc.weight']=folded.to(torch.bfloat16).contiguous()
    out=a.out/'initializer/export';out.mkdir(parents=True,exist_ok=True)
    save_file(state,str(out/'model.safetensors'))
    config=json.loads((Path(reference['base_export'])/'config.json').read_text())
    # Native drafter parameters stay unchanged; only new target taps differ.
    config['dflash_config']['target_layer_ids']=[1,8,15,22,29]
    config['num_target_layers']=32
    write(out/'config.json',config)
    rows=json.loads((a.out/'train.json').read_text())
    write(a.out/'transfer.json',{'status':'complete','family':'cross','target_adapters':None,
        'input_width':20480,'draft':reference['draft'],'base_export':str(out),
        'base_sha256':digest(out/'model.safetensors'),'base_training_epochs':len(metadata['history']),
        'records':len(rows),'map_checkpoint':str(checkpoint),'map_checkpoint_format':'archived_cross_pilot',
        'map_checkpoint_sha256':digest(checkpoint),'initializer_records':512,
        'initializer_scope':metadata['scope'],'scope':'integration pilot only; not a matched full-data experiment',
        'target_taps':[1,8,15,22,29],'training_manifest_sha256':digest(a.out/'train.json')})
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['archived','reference','out']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
