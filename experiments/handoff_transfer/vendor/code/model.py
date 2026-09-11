"""Matched data and CE for learned bias, fusion LoRA and their combination."""
import argparse, time, math, random
from common import *
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

VARIANTS = ['fusion_r56']

class Steering(nn.Module):
    def __init__(self,fc):
        super().__init__();self.fc=fc;self.bias=nn.Parameter(torch.zeros(2560,dtype=torch.float32))
    def forward(self,x):
        y=self.fc(x);return (y.float()+self.bias).to(y.dtype)

def build(kind):
    DFlash, Online, _ = load_upstream()
    from transformers import Qwen3Config
    from peft import LoraConfig, inject_adapter_in_model
    cfg = Qwen3Config.from_pretrained(PAIR/'draft')
    cfg._attn_implementation = 'sdpa'
    draft = DFlash(cfg).to(dtype=torch.bfloat16)
    draft.load_state_dict(load_file(PAIR/'draft/model.safetensors'), strict=True)
    draft.requires_grad_(False)
    if kind != 'bias':
        assert kind in VARIANTS
        rank = 56
        targets = ['fc']
        inject_adapter_in_model(LoraConfig(r=rank,lora_alpha=rank,lora_dropout=0,
                                          bias='none',target_modules=targets), draft)
        for p in draft.parameters():
            if p.requires_grad: p.data = p.data.float()
    if kind in ['bias','bias_fusion']:draft.fc=Steering(draft.fc)
    idx = json.loads((PAIR/'target/model.safetensors.index.json').read_text())['weight_map']
    with safe_open(PAIR/'target'/idx['model.embed_tokens.weight'],framework='pt') as f:
        weight = f.get_tensor('model.embed_tokens.weight')
    embed = nn.Embedding.from_pretrained(weight,freeze=True)
    head = nn.Linear(weight.shape[1],weight.shape[0],bias=False,dtype=weight.dtype)
    head.weight = embed.weight
    wrapper = Online(draft,head,embed,mask_token_id=151669,block_size=16,
                     attention_backend='sdpa',num_anchors=8,loss_decay_gamma=None,
                     objective_chunk_blocks=2,loss_type='dflash')
    from objectives import configure
    configure(wrapper,'auf')
    return wrapper.cuda(), cfg

def export(model,cfg,kind,dst):
    from peft.tuners.lora.layer import LoraLayer
    draft = model.draft_model
    projection=draft.fc
    bias=projection.bias if isinstance(projection,Steering) else None
    if bias is not None:draft.fc=projection.fc
    try:
        state={k.replace('.base_layer',''):v.detach().to(dtype=torch.bfloat16,device='cpu').contiguous() for k,v in draft.state_dict().items() if '.lora_' not in k}
        for name,mod in draft.named_modules():
            if isinstance(mod,LoraLayer):
                state[name+'.weight']=(mod.base_layer.weight.float()+mod.get_delta_weight('default').float()).to(dtype=torch.bfloat16,device='cpu').contiguous()
    finally:draft.fc=projection
    dst.mkdir(parents=True,exist_ok=True)
    if bias is not None:torch.save(bias.detach().cpu(),dst/'context_bias.pt')
    save_file(state,str(dst/'model.safetensors'))
    (dst/'config.json').write_text((PAIR/'draft/config.json').read_text())

def shard_batches(tag,epoch):
    paths = sorted((PREV/'features'/tag).glob('*.pt'))
    rng=random.Random(42+epoch);rng.shuffle(paths)
    for path in paths:
        s=torch.load(path,weights_only=True,mmap=True)
        ii=list(range(len(s['rows'])));rng.shuffle(ii)
        ii.sort(key=lambda i:len(s['rows'][i]['full_ids'])//64)
        for start in range(0,len(ii),2):
            indices=ii[start:start+2];rows=[s['rows'][i] for i in indices]
            S=max(len(x['full_ids']) for x in rows)
            ids=torch.zeros((len(rows),S),dtype=torch.long)
            h=torch.zeros((len(rows),S,12800),dtype=torch.bfloat16)
            mask=torch.zeros((len(rows),S))
            for j,(i,x) in enumerate(zip(indices,rows)):
                n=len(x['full_ids']);ids[j,:n]=torch.tensor(x['full_ids'])
                h[j,:n]=s['features'][i];mask[j,len(x['prompt_token_ids']):n]=1
            yield ids.cuda(),h.cuda(),mask.cuda()
