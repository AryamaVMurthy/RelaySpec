class MetricsWorker:
    def sd_verify_draft_projections(self,export_path):
        """Verify merged drafter weights and vLLM's derived context-KV cache."""
        from pathlib import Path
        import torch
        from safetensors import safe_open
        draft=self.model_runner.get_draft_model() if hasattr(self.model_runner,'get_draft_model') else self.model_runner.drafter.model
        count=0;kv=[]
        with safe_open(Path(export_path)/'model.safetensors',framework='pt',device='cpu') as reader:
            for index,layer in enumerate(draft.model.layers):
                prefix=f'layers.{index}.'
                expected=[reader.get_tensor(prefix+'self_attn.'+name+'.weight') for name in ['q_proj','k_proj','v_proj']]
                packed=torch.cat(expected,dim=0)
                torch.testing.assert_close(layer.self_attn.qkv_proj.weight.detach().cpu(),packed,rtol=0,atol=0)
                count+=packed.numel();kv.extend(expected[1:])
                for source,actual in [('self_attn.o_proj',layer.self_attn.o_proj.weight),
                                      ('mlp.down_proj',layer.mlp.down_proj.weight)]:
                    value=reader.get_tensor(prefix+source+'.weight')
                    torch.testing.assert_close(actual.detach().cpu(),value,rtol=0,atol=0)
                    count+=value.numel()
                packed=torch.cat([reader.get_tensor(prefix+'mlp.'+name+'.weight') for name in ['gate_proj','up_proj']],dim=0)
                torch.testing.assert_close(layer.mlp.gate_up_proj.weight.detach().cpu(),packed,rtol=0,atol=0)
                count+=packed.numel()
        assert count>0
        assert hasattr(draft.model,'_fused_kv_weight'),'Expected pinned vLLM context-KV buffer'
        torch.testing.assert_close(draft.model._fused_kv_weight.detach().cpu(),torch.cat(kv,dim=0),rtol=0,atol=0)
        return {'passed':True,'projection_elements_verified':count,'full_tensor_equality':True,'fused_context_kv_verified':True}

    def sd_install_monitor(self):
        from vllm.utils import jit_monitor
        if not hasattr(jit_monitor,"sd_events"):
            jit_monitor.sd_events=0
            old=jit_monitor._handle_jit_event
            def counted(**kwargs):
                jit_monitor.sd_events+=1
                return old(**kwargs)
            jit_monitor._handle_jit_event=counted
        return True

    def sd_stats(self):
        import torch
        from vllm.utils import jit_monitor
        d=getattr(self.model_runner,"speculator",None) or getattr(self.model_runner,"drafter",None)
        t=getattr(d,"transfer_teacher",None)
        return dict(jit_events=getattr(jit_monitor,"sd_events",0),device_name=torch.cuda.get_device_name(),device_uuid=str(torch.cuda.get_device_properties(0).uuid),allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved(), teacher_graph_captures=getattr(t,"graph_captures",0), teacher_eager_prefills=getattr(t,"eager_prefills",0))

    def sd_verify_attachments(self,export_path,target_size):
        from pathlib import Path
        import json,torch
        from safetensors import safe_open
        from data import tensor
        def sample(x):
            rows=torch.tensor([0,151645,151669,151935],device=x.device)
            cols=torch.tensor([0,1,x.shape[1]//2,x.shape[1]-1],device=x.device)
            return x.index_select(0,rows).index_select(1,cols).detach().cpu()
        target=self.model_runner.model
        expected=tensor(None,target_size,'model.embed_tokens.weight')
        torch.testing.assert_close(sample(target.model.embed_tokens.weight),sample(expected),rtol=0,atol=0)
        expected=tensor(None,target_size,'lm_head.weight') if target_size==8 else expected
        torch.testing.assert_close(sample(target.lm_head.weight),sample(expected),rtol=0,atol=0)
        if export_path:
            if hasattr(self.model_runner,"get_draft_model"):
                draft=self.model_runner.get_draft_model()
            else:
                draft=self.model_runner.drafter.model
            assert draft is not None
            root=Path(export_path)
            for key,actual in [('embed_tokens.weight',draft.model.embed_tokens.weight),('lm_head.weight',draft.lm_head.weight)]:
                with safe_open(root/'model.safetensors',framework='pt') as f:expected=f.get_tensor(key)
                assert actual.shape[1]==expected.shape[1]
                torch.testing.assert_close(sample(actual),sample(expected),rtol=0,atol=0)
            # Verify the learned interface too; a source fc must never survive
            # in place of the exported target-width folded matrix.
            with safe_open(root/'model.safetensors',framework='pt') as f:
                expected_fc=f.get_tensor('fc.weight')
                expected_norm=f.get_tensor('hidden_norm.weight')
            actual_fc=draft.model.fc.weight
            assert actual_fc.shape == expected_fc.shape
            r=torch.tensor([0,actual_fc.shape[0]//2,actual_fc.shape[0]-1])
            c=torch.tensor([0,1,actual_fc.shape[1]//2,actual_fc.shape[1]-1])
            actual_sample=actual_fc.index_select(0,r.to(actual_fc.device)).index_select(1,c.to(actual_fc.device)).cpu()
            torch.testing.assert_close(actual_sample,expected_fc[r][:,c],rtol=0,atol=0)
            torch.testing.assert_close(draft.model.hidden_norm.weight.cpu(),expected_norm,rtol=0,atol=0)
        projections=self.sd_verify_draft_projections(export_path) if export_path else None
        return {'passed':True,'target_size':target_size,'target_and_draft_embedding_head_samples_verified':True,
                'draft_projections':projections}
