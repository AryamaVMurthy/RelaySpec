import json,torch
from safetensors.torch import load_file,save_file
from paths import WORK,model,put
from train import make,SPEC
from data import tensor,pairs
from mapper import relative
@torch.no_grad()
def main():
 out=WORK/'export';assert not out.exists(),'Do not overwrite an export'
 q=torch.load(WORK/'fit/final.pt',weights_only=True)
 assert q['spec']==SPEC and q['epoch']==2 and q['selection']=='final_fixed_epoch'
 m=make().cuda();m.load_state_dict(q['model']);m.eval()
 weights=load_file(model(4,'draft')/'model.safetensors');cfg=json.loads((model(4,'draft')/'config.json').read_text())
 fc=m.folded().to(torch.bfloat16)
 _,x,*_=next(pairs('train',1,4096));x=x[:256].cuda()
 with torch.autocast('cuda',dtype=torch.bfloat16):
  expected=m(x)[0];actual=m.frozen_norm(torch.nn.functional.linear(x,fc));error=relative(actual,expected).mean().item()
 assert error<1e-3,error
 e4=tensor(None,4,'model.embed_tokens.weight')
 weights['embed_tokens.weight']=e4.clone();weights['lm_head.weight']=e4.clone();weights['fc.weight']=fc.cpu().contiguous()
 cfg.update(target_hidden_size=4096,architectures=['MapperDFlash'],tie_word_embeddings=False)
 out.mkdir(parents=True)
 # Storage packaging only: identical tensor values in a single portable file.
 save_file({k:v.contiguous() for k,v in weights.items()},str(out/'model.safetensors'))
 put(out/'config.json',cfg);put(out/'export_check.json',{'context_relative_mse':error,'checkpoint_epoch':2})
if __name__=='__main__':main()
