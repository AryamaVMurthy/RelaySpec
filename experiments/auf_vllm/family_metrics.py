"""Runtime asset checks for original-target, same-family vLLM evaluations."""
from .runtime_zip.metrics import MetricsWorker


class FamilyMetricsWorker(MetricsWorker):
    def sd_verify_family_assets(self,target_path,export_path=None,native_path=None):
        import json
        from pathlib import Path
        import torch
        from safetensors import safe_open

        def tensor(root,key):
            root=Path(root)
            index=root/'model.safetensors.index.json'
            paths=[root/json.loads(index.read_text())['weight_map'][key]] if index.exists() else list(root.glob('*.safetensors'))
            for path in paths:
                with safe_open(path,framework='pt',device='cpu') as reader:
                    if key in reader.keys():
                        return reader.get_tensor(key)
            raise KeyError(key)

        def check(actual,expected):
            assert tuple(actual.shape)==tuple(expected.shape),(actual.shape,expected.shape)
            if actual.ndim==1:
                torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)
            else:
                rows=torch.tensor([0,1,actual.shape[0]//2,actual.shape[0]-1])
                cols=torch.tensor([0,1,actual.shape[1]//2,actual.shape[1]-1])
                sample=actual.index_select(0,rows.to(actual.device)).index_select(1,cols.to(actual.device)).cpu()
                torch.testing.assert_close(sample,expected[rows][:,cols],rtol=0,atol=0)

        target=self.model_runner.model
        config=json.loads((Path(target_path)/'config.json').read_text())
        embedding=tensor(target_path,'model.embed_tokens.weight')
        check(target.model.embed_tokens.weight,embedding)
        head=embedding if config.get('tie_word_embeddings',False) else tensor(target_path,'lm_head.weight')
        check(target.lm_head.weight,head)
        checks=['target_embedding','target_head']
        normal_projection=None
        if export_path:
            draft=self.model_runner.get_draft_model() if hasattr(self.model_runner,'get_draft_model') else self.model_runner.drafter.model
            assert draft is not None
            for key,actual in [('embed_tokens.weight',draft.model.embed_tokens.weight),
                               ('lm_head.weight',draft.lm_head.weight),('fc.weight',draft.model.fc.weight),
                               ('hidden_norm.weight',draft.model.hidden_norm.weight)]:
                check(actual,tensor(export_path,key));checks.append('draft_'+key)
            export_config=json.loads((Path(export_path)/'config.json').read_text())
            if 'relayspec_normal_input_eps' in export_config:
                import torch.nn.functional as F
                weight=draft.model.fc.weight
                x=torch.sin(torch.arange(3*weight.shape[1],device=weight.device,dtype=torch.float32)).reshape(3,-1)
                x=(x*torch.tensor([.01,1.,100.],device=x.device)[:,None]).to(weight.dtype)
                with torch.no_grad():
                    actual=draft.combine_hidden_states(x)
                    expected=F.linear(F.rms_norm(x,(x.shape[-1],),eps=export_config['relayspec_normal_input_eps']),weight)
                    raw=F.linear(x,weight)
                error=float((actual.float()-expected.float()).square().sum()/expected.float().square().sum().clamp_min(1e-12))
                raw_error=float((actual.float()-raw.float()).square().sum()/expected.float().square().sum().clamp_min(1e-12))
                assert error<1e-3 and raw_error>.1,(error,raw_error)
                normal_projection={'normalized_reference_relative_mse':error,'unnormalized_negative_control_relative_mse':raw_error}
                checks.append('normal_input_rms_before_projection')
        if native_path:
            draft=self.model_runner.get_draft_model() if hasattr(self.model_runner,'get_draft_model') else self.model_runner.drafter.model
            check(draft.model.embed_tokens.weight,embedding)
            check(draft.lm_head.weight,head)
            for key,actual in [('fc.weight',draft.model.fc.weight),('hidden_norm.weight',draft.model.hidden_norm.weight)]:
                check(actual,tensor(native_path,key))
            checks.extend(['native_embedding','native_head','native_fc','native_norm'])
        projections=self.sd_verify_draft_projections(export_path) if export_path else None
        return {'passed':True,'checks':checks,'verification':'matrix shape and deterministic samples; full norm vector',
                'draft_projections':projections,'normal_projection':normal_projection,
                'target_adapters':None}
