"""Twenty-epoch optimization control, reuses captured pilot features."""
from train_zip_family_pilot import *

def main():
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device(f'cuda:{rank}'))
    if rank in (0,2):
        pair='llama' if rank==0 else 'cross'
        cache_root=Path('/scratch/aryama.murthy/zip-family-pilot-20260907')/pair
        root=cache_root.parent/(pair+'-epoch20')
        root.mkdir(parents=True,exist_ok=True)
        q=torch.load(cache_root/'maps.pt',weights_only=True)
        cfg=q['config']; fusion=q['model']['fusion']; norm=q['model']['norm']
        do,di=q['model']['maps.0.weight'].shape
        target_taps=tuple(cfg['relay_training']['target_layer_ids'])
        source_taps=tuple(cfg['source_trunk']['tap_layers'])
        manifest=Path('configs/train_math_4096.json')
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        records = sum([torch.load(cache_root / f"features-{j}.pt", weights_only=True) for j in range(2)], [])
        records.sort(key=lambda r: r["row"])
        assert [r["row"] for r in records] == list(range(512))
        x = torch.cat([r["x"] for r in records]); y = torch.cat([r["y"] for r in records])
        w = torch.cat([torch.full((len(r["x"]),), 1/len(r["x"])) for r in records])
        del records
        m = Context(fusion, norm, d8=di, d4=do, layers=5).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=0, fused=True)
        batch = 2048; steps = math.ceil(len(x)/batch)*20; warm = max(1, int(.05*steps)); step = 0
        history = []; started = time.perf_counter()
        for epoch in range(20):
            order = torch.randperm(len(x), generator=torch.Generator().manual_seed(42+epoch))
            total = 0.; epoch_start = time.perf_counter()
            for inds in order.split(batch):
                rate = (step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,steps-warm)))
                opt.param_groups[0]["lr"] = .001*rate
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = m.loss(x[inds].cuda(), y[inds].cuda())
                weighted = (losses*w[inds].cuda()).sum()
                loss = weighted*(len(x)/512)/batch
                assert torch.isfinite(loss)
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True); opt.step()
                total += weighted.item(); step += 1
            event = {"epoch": epoch+1, "loss": total/512, "seconds": time.perf_counter()-epoch_start}
            history.append(event); print(json.dumps({"pair": pair, **event}), flush=True)
        folded = m.folded().to(torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            sample = x[:256].cuda(); a, _ = m(sample)
            b = m.frozen_norm(torch.nn.functional.linear(sample, folded))
            fold_error = ((a.float()-b.float()).square().sum(-1)/(a.float().square().sum(-1)+1e-6)).mean().item()
        assert fold_error < .001, fold_error
        torch.save({"model": {k:v.cpu() for k,v in m.state_dict().items()}, "config": cfg}, root/"maps.pt")
        torch.save({"relay": {"projection.weight": folded.cpu()}, "target_layer_ids": target_taps, "source_layer_ids": source_taps, "relay_architecture": "raw_linear", "proposer_family": "dflash", "feature_objective": "zip_layer_plus_context", "steps": step}, root/"relay.pt")
        summary = {"pair": pair, "records":512, "positions":len(x), "history":history, "train_seconds":time.perf_counter()-started, "fold_relative_mse":fold_error, "alignment":"identical plain text, exact shared token-end boundaries", "max_input_tokens":1024, "sample_fraction":.25, "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(), "job_id":os.environ["SLURM_JOB_ID"], "config":cfg, "scope":"ZIP objective and optimizer pilot; existing solution text, not generated rollout reproduction"}
        (root/"summary.json").write_text(json.dumps(summary, indent=2))
    dist.barrier(); dist.destroy_process_group()

if __name__=='__main__': main()
