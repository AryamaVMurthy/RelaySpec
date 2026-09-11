"""Family-dependent interface only; training/objective remain the handoff recipe."""
import argparse,time,math,random
from common import *
import torch
from torch import nn
from safetensors.torch import load_file,save_file

TRANSFER=json.loads((R/'transfer.json').read_text())
assert TRANSFER['status']=='complete' and TRANSFER['target_adapters'] is None
INPUT_WIDTH=TRANSFER['input_width']
from five_maps import FiveMapProjection
VARIANTS=['fusion_r56','five_maps']

def build(kind):
    assert kind in VARIANTS
    DFlash,Online,_=load_upstream()
    from transformers import Qwen3Config
    from peft import LoraConfig,inject_adapter_in_model
    native=Path(TRANSFER['draft']);base=Path(TRANSFER['base_export'])
    assert sha(base/'model.safetensors')==TRANSFER['base_sha256']
    cfg=Qwen3Config.from_pretrained(native,local_files_only=True);cfg._attn_implementation='sdpa'
    draft=DFlash(cfg).to(dtype=torch.bfloat16)
    native_state=load_file(str(native/'model.safetensors'))
    draft.load_state_dict(native_state,strict=True)
    base_state=load_file(str(base/'model.safetensors'))
    assert all(torch.equal(value,base_state[key]) for key,value in native_state.items() if key!='fc.weight')
    f0=base_state['fc.weight'];width=cfg.hidden_size
    assert f0.shape==(width,INPUT_WIDTH)
    draft.requires_grad_(False)
    initialization={'base_export_sha256':TRANSFER['base_sha256']}
    if kind=='fusion_r56':
        # Meta allocation preserves the packaged PEFT random stream.
        with torch.device('meta'):projection=nn.Linear(INPUT_WIDTH,width,bias=False,dtype=torch.bfloat16)
        projection.weight=nn.Parameter(f0.clone(),requires_grad=False)
        draft.fc=projection
        inject_adapter_in_model(LoraConfig(r=56,lora_alpha=56,lora_dropout=0,bias='none',target_modules=['fc']),draft)
    else:
        checkpoint=base.parents[1]/'resume.pt'
        saved=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
        assert saved['epoch']==2, 'ZIP maps must be from the epoch-3 export'
        maps=saved['mapper']
        assert torch.equal(maps['fusion'],native_state['fc.weight'])
        assert torch.equal(maps['norm'],native_state['hidden_norm.weight'])
        draft.fc=FiveMapProjection(native_state['fc.weight'],[maps[f'maps.{i}.weight'] for i in range(5)])
        assert draft.fc.target_width*5==INPUT_WIDTH
        folded=draft.fc.folded().detach().to(torch.bfloat16)
        error=float((folded.float()-f0.float()).square().sum()/f0.float().square().sum().clamp_min(1e-12))
        assert error<1e-6, f'ZIP checkpoint/export mismatch: {error}'
        initialization.update({'map_checkpoint':str(checkpoint),'map_checkpoint_sha256':sha(checkpoint),
                               'folded_initial_relative_mse':error})
    for p in draft.parameters():
        if p.requires_grad:p.data=p.data.float()
    embedding=base_state['embed_tokens.weight'];head_weight=base_state['lm_head.weight']
    assert embedding.shape==head_weight.shape and embedding.shape[1]==width
    embed=nn.Embedding.from_pretrained(embedding,freeze=True)
    head=nn.Linear(width,head_weight.shape[0],bias=False,dtype=head_weight.dtype)
    head.weight=nn.Parameter(head_weight,requires_grad=False)
    if TRANSFER['family']!='llama':
        assert torch.equal(embedding,head_weight);head.weight=embed.weight
    block,mask=draft.block_size,draft.mask_token_id
    assert isinstance(mask,int) and 0<=mask<embedding.shape[0]
    wrapper=Online(draft,head,embed,mask_token_id=mask,block_size=block,attention_backend='sdpa',
                   num_anchors=8,loss_decay_gamma=None,objective_chunk_blocks=2,loss_type='dflash')
    from objectives import configure
    configure(wrapper,'auf')
    n=sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    assert n==(56*(INPUT_WIDTH+width) if kind=='fusion_r56' else INPUT_WIDTH*width)
    if int(os.environ.get('LOCAL_RANK','0'))==0:put(R/f'model-contract-{kind}.json',{'family':TRANSFER['family'],'base_sha256':TRANSFER['base_sha256'],
        'input_width':INPUT_WIDTH,'source_width':width,'trainable_parameters':n,'variant':kind,'rank':56 if kind=='fusion_r56' else None,'alpha':56 if kind=='fusion_r56' else None,
        'block_size':block,'mask_token_id':mask,'num_anchors':8,'objective_chunk_blocks':2,
        'target_adapters':None,'initialization':initialization,'base_training_epochs':TRANSFER['base_training_epochs'],
        'loss_source_sha256':sha(CODE/'objectives.py'),'source_head_untied':TRANSFER['family']=='llama'})
    return wrapper.cuda(),cfg

def export(model,cfg,kind,dst):
    from peft.tuners.lora.layer import LoraLayer
    draft=model.draft_model
    state={k.replace('.base_layer',''):v.detach().to(dtype=torch.bfloat16,device='cpu').contiguous()
           for k,v in draft.state_dict().items() if '.lora_' not in k and not (kind=='five_maps' and k.startswith('fc.'))}
    if kind=='five_maps':state['fc.weight']=draft.fc.folded().detach().to(dtype=torch.bfloat16,device='cpu').contiguous()
    for name,module in draft.named_modules():
        if isinstance(module,LoraLayer):
            state[name+'.weight']=(module.base_layer.weight.float()+module.get_delta_weight('default').float()).to(dtype=torch.bfloat16,device='cpu').contiguous()
    state['embed_tokens.weight']=model.embed_tokens.weight.detach().cpu().contiguous()
    state['lm_head.weight']=model.lm_head.weight.detach().cpu().contiguous()
    dst.mkdir(parents=True,exist_ok=True);save_file(state,str(dst/'model.safetensors'))
    # Preserve the already-audited transfer runtime dimensions, target taps and heads.
    (dst/'config.json').write_text((Path(TRANSFER['base_export'])/'config.json').read_text())
