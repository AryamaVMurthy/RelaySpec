"""Streaming same-vocabulary Qwen14/Llama training; frozen original targets."""
import argparse
import importlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from .blocks import make_block, collate_blocks, conditioned_forward
from .interfaces import LayerContextMapper
from .losses import token_loss
from .pilot_data import sha, write
from .runtime_zip.sampling import positions
from .train_pilot import tensor


class FamilyRecords:
    def __init__(self, root, count=None, split="train"):
        manifests = [root/f"{split}.json"] if (root/f"{split}.json").exists() else sorted(root.glob(f"part-*/{split}.json"), key=lambda p: int(p.parent.name.split('-')[-1]))
        self.items = []
        self.manifests = {str(p): sha(p) for p in manifests}
        for path in manifests:
            for index, row in enumerate(json.loads(path.read_text())):
                self.items.append((path.parent, split, index, row))
        if count is not None:
            assert len(self.items) >= count, (len(self.items), count)
            self.items = self.items[:count]
        self.rows = [item[-1] for item in self.items]
        assert self.rows and len({r['group_id'] for r in self.rows}) == len(self.rows)

    def __len__(self):
        return len(self.items)

    def get(self, index, source=False):
        root, split, local, row = self.items[index]
        captured = []
        for role in (["target", "source"] if source else ["target"]):
            path = root/f"features/{role}/{split}/{local:05d}.pt"
            metadata = json.loads(path.with_suffix('.json').read_text())
            assert metadata['manifest_sha256'] == self.manifests[str(root/f'{split}.json')]
            value = torch.load(path, weights_only=True)
            assert value['group_id'] == row['group_id']
            assert len(value['features']) == len(row['full_ids'])
            captured.append(value['features'])
        return row, *captured


class FamilyRuntime:
    def __init__(self, models, family):
        from transformers import Qwen3Config
        is_llama = family == 'llama'
        source, target, draft = ('llama8-source', 'llama3-target', 'llama8-draft') if is_llama else ('qwen4-source', 'qwen14-target', 'qwen4-draft')
        self.source_path, self.target_path, self.draft_path = (models/x for x in [source, target, draft])
        self.source_width, self.target_width = (4096, 3072) if is_llama else (2560, 5120)
        self.taps = [1,7,13,19,25] if is_llama else [1,10,19,28,37]
        self.target_config = json.loads((self.target_path/'config.json').read_text())
        eos = self.target_config['eos_token_id']
        self.eos = tuple(eos) if isinstance(eos, list) else (eos,)
        sys.path.insert(0, os.environ['DFLASH_SOURCE'])
        cls = importlib.import_module('dflash.model').DFlashDraftModel
        cfg = Qwen3Config.from_pretrained(self.draft_path, local_files_only=True)
        cfg._attn_implementation = 'sdpa'
        self.draft = cls(cfg)
        self.draft.load_state_dict(load_file(str(self.draft_path/'model.safetensors')), strict=True, assign=True)
        self.draft = self.draft.to('cuda', torch.bfloat16).eval().requires_grad_(False)
        self.embedding = tensor(self.source_path, 'model.embed_tokens.weight').to('cuda', torch.bfloat16)
        self.head = tensor(self.source_path, 'lm_head.weight').to('cuda', torch.bfloat16) if is_llama else self.embedding
        assert self.embedding.shape == self.head.shape
        assert self.embedding.shape[1] == self.source_width
        self.untied = is_llama
        if is_llama:
            assert not torch.equal(self.embedding[:8], self.head[:8])

    def block(self, row, features, anchor):
        assert features.shape[-1] == 5*self.target_width
        return make_block(features, row['full_ids'], anchor, self.draft.block_size, self.draft.mask_token_id, self.eos)

    def logits(self, mapper, batch):
        context, _ = mapper(batch['context'])
        noise = F.embedding(batch['noise_ids'], self.embedding)
        hidden = conditioned_forward(self.draft, context, noise, batch['position_ids'], batch['attention_mask'])
        return F.linear(hidden, self.head)

    @torch.no_grad()
    def export(self, mapper, dest, features):
        folded = mapper.folded().to(torch.bfloat16)
        x = features[:256].to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            a, b = mapper(x)[0], mapper.normalize(F.linear(x, folded))
        error = ((a.float()-b.float()).square().sum(-1)/(a.float().square().sum(-1)+1e-6)).mean().item()
        assert error < 1e-3, error
        weights = load_file(str(self.draft_path/'model.safetensors'))
        weights.update({'fc.weight': folded.cpu().contiguous(), 'embed_tokens.weight': self.embedding.cpu().clone(), 'lm_head.weight': self.head.cpu().clone()})
        dest.mkdir(parents=True, exist_ok=True)
        save_file(weights, str(dest/'model.safetensors'))
        config = json.loads((self.draft_path/'config.json').read_text())
        config.update(architectures=['MapperDFlash'], target_hidden_size=self.target_width,
                      num_target_layers=self.target_config['num_hidden_layers'], tie_word_embeddings=False,
                      bos_token_id=self.target_config['bos_token_id'], eos_token_id=self.target_config['eos_token_id'])
        config['dflash_config']['target_layer_ids'] = self.taps
        write(dest/'config.json', config)
        write(dest/'export_check.json', {'context_relative_mse': error, 'source_head_untied': self.untied})


def paired_batches(records, seed, batch_size=2048):
    generator = torch.Generator().manual_seed(seed)
    xs, ys, ws, count = [], [], [], 0
    for index in range(len(records)):
        row, x, y = records.get(index, source=True)
        pos = positions(len(x), len(row['prompt_token_ids']), row['group_id'])
        order = torch.randperm(len(pos), generator=generator)
        selected = torch.tensor(pos)[order]
        xs.append(x[selected]); ys.append(y[selected]); ws.append(torch.full((len(pos),), 1/len(pos)))
        count += len(pos)
        if count >= batch_size:
            x, y, w = torch.cat(xs), torch.cat(ys), torch.cat(ws)
            while len(x) >= batch_size:
                yield x[:batch_size], y[:batch_size], w[:batch_size]
                x, y, w = x[batch_size:], y[batch_size:], w[batch_size:]
            xs, ys, ws, count = [x], [y], [w], len(x)
    if count:
        yield torch.cat(xs), torch.cat(ys), torch.cat(ws)


def main(args):
    records = FamilyRecords(args.data, args.records)
    validation = json.loads(args.validation_manifest.read_text())
    assert not ({r['group_id'] for r in records.rows} & {r['group_id'] for r in validation})
    assert args.exploratory or len(validation) >= 1024
    args.out.mkdir(parents=True, exist_ok=True)
    contract = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    contract.update(manifests=records.manifests, validation_sha256=sha(args.validation_manifest), target_adapters=None,
                    token_batch='4 records x 4 anchors x 8 accumulation', zip_positions='deterministic stratified 25%; equal example mass; 2048-position batches')
    if (args.out/'contract.json').exists():
        assert json.loads((args.out/'contract.json').read_text()) == contract
    else:
        write(args.out/'contract.json', contract)
    runtime = FamilyRuntime(args.models, args.family)
    torch.manual_seed(args.seed)
    mapper = LayerContextMapper(runtime.draft.fc.weight, runtime.draft.hidden_norm.weight,
                                target_width=runtime.target_width, source_width=runtime.source_width).to('cuda')
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=args.lr, weight_decay=0, fused=True)
    total_positions = sum(len(positions(len(r['full_ids']),len(r['prompt_token_ids']),r['group_id'])) for r in records.rows)
    steps_epoch = math.ceil(total_positions/2048) if args.objective == 'zip' else math.ceil(len(records)/32)
    total_steps, warmup = steps_epoch*args.epochs, max(1, int(.05*steps_epoch*args.epochs))
    step, first_epoch, history = 0, 0, []
    resume = args.out/'resume.pt'
    if resume.exists():
        state = torch.load(resume, weights_only=True)
        assert state['contract'] == contract
        mapper.load_state_dict(state['mapper']); optimizer.load_state_dict(state['optimizer'])
        step, first_epoch, history = state['step'], state['epoch']+1, state['history']
    frozen = {name: p._version for name, p in runtime.draft.named_parameters()}
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(first_epoch, args.epochs):
        start, active, valid, blocks, loss_sum = time.perf_counter(), 0, 0, 0, 0.
        rng = random.Random(args.seed+epoch)
        order = list(range(len(records))); rng.shuffle(order)
        iterator = paired_batches(records, args.seed+epoch) if args.objective == 'zip' else (order[i:i+32] for i in range(0,len(order),32))
        for iteration, item in enumerate(iterator):
            rate = (step+1)/warmup if step < warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
            for group in optimizer.param_groups:
                group['lr'] = args.lr*rate
            optimizer.zero_grad(set_to_none=True)
            if args.objective == 'zip':
                x,y,w = item
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    weighted = (mapper.feature_loss(x.to('cuda'),y.to('cuda'))*w.to('cuda')).sum()
                    loss = weighted*(total_positions/len(records))/2048
                loss.backward(); loss_sum += weighted.item()/len(records)
            else:
                micro_count = math.ceil(len(item)/4)
                for offset in range(0,len(item),4):
                    batch_blocks = []
                    for index in item[offset:offset+4]:
                        row, x = records.get(index)
                        candidates = range(len(row['prompt_token_ids']), len(row['full_ids'])-1)
                        assert candidates, f'No supervised positions: {index}'
                        for anchor in rng.sample(candidates, min(4,len(candidates))):
                            batch_blocks.append(runtime.block(row,x,anchor))
                    batch = collate_blocks(batch_blocks, 'cuda')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        scores = runtime.logits(mapper,batch)
                        result = token_loss(scores,batch['labels'],batch['valid'],args.objective)
                    (result.loss/micro_count).backward()
                    active += int(result.active.sum()); valid += int(batch['valid'].sum()); blocks += len(batch_blocks)
                    loss_sum += result.loss.item()/micro_count/steps_epoch
                    del scores, result, batch, batch_blocks
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); step += 1
            if (iteration+1) % 16 == 0:
                print(json.dumps({'epoch':epoch+1,'step':step,'seconds':time.perf_counter()-start}),flush=True)
        assert iteration+1 == steps_epoch
        assert all(p.grad is None and p._version == frozen[name] for name,p in runtime.draft.named_parameters())
        event = {'epoch':epoch+1,'steps':step,'records':len(records),'blocks':blocks,'active_tokens':active,
                 'valid_tokens':valid,'training_loss':loss_sum,'seconds':time.perf_counter()-start}
        history.append(event)
        temporary = resume.with_suffix('.part')
        torch.save({'contract':contract,'mapper':mapper.state_dict(),'optimizer':optimizer.state_dict(),
                    'epoch':epoch,'step':step,'history':history},temporary)
        temporary.replace(resume)
        runtime.export(mapper,args.out/f'epoch-{epoch+1}/export',records.get(0)[1])
        write(args.out/'history.json',history)
        print(json.dumps(event),flush=True)
    write(args.out/'summary.json',{'contract':contract,'history':history,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                 'checkpoint_sha256':sha(resume),'job_id':os.environ.get('SLURM_JOB_ID'),
                                 'status':'fit_complete_offline_validation_pending'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=['llama','q14'], required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--validation-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--objective', choices=['zip','ce','auf'], required=True)
    parser.add_argument('--records', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--exploratory', action='store_true')
    main(parser.parse_args())
