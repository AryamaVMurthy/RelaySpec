import torch

from core import ProgressHead, progress_loss, stop_aware_prefix_mask


def test_progress_head_starts_as_exact_identity():
    hidden = torch.randn(2, 15, 8)
    head = ProgressHead(8, 2)
    assert torch.equal(hidden, head(hidden))


def test_invalid_post_rejection_labels_have_zero_gradient():
    logits = torch.randn(1, 5, 7, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    native = torch.tensor([[1, 2, 0, 0, 0]])
    valid = torch.tensor([[True, True, True, False, False]])
    loss, metrics = progress_loss(logits, labels, valid, native, error_weight=4.)
    loss.backward()
    assert logits.grad[:, :3].abs().sum() > 0
    assert logits.grad[:, 3:].abs().sum() == 0
    assert metrics["first_errors"] == 1
    changed = labels.clone()
    changed[:, 3:] = 0
    other, _ = progress_loss(logits.detach(), changed, valid, native, error_weight=4.)
    assert torch.equal(loss.detach(), other)


def test_position_specific_correction_preserves_other_positions_exactly():
    hidden = torch.ones(2, 15, 8)
    head = ProgressHead(8, 2, positions=[0])
    with torch.no_grad():
        head.up.weight.fill_(1)
        head.down.weight.fill_(1)
    output = head(hidden)
    assert torch.equal(output[:, 1:], hidden[:, 1:])
    assert not torch.equal(output[:, 0], hidden[:, 0])


def test_margin_preservation_loss_detaches_teacher_and_masks_future():
    student = torch.randn(1, 5, 7, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    native = torch.tensor([[1, 2, 0, 0, 0]])
    valid = torch.tensor([[True, True, True, False, False]])
    loss, _ = progress_loss(student, labels, valid, native, kind="margin_kl", native_logits=teacher)
    loss.backward()
    assert teacher.grad is None
    assert student.grad[:, 3:].abs().sum() == 0


def test_eos_label_is_kept_but_later_positions_are_masked():
    native = torch.tensor([[1, 99, 2, 3], [1, 2, 3, 4]])
    valid = torch.ones_like(native, dtype=torch.bool)
    assert stop_aware_prefix_mask(valid, native, [99]).tolist() == [[True, True, False, False], [True, True, True, True]]


def test_soft_progress_rewards_extending_prefix_and_masks_unverified_tail():
    logits = torch.tensor([[[5., 0.], [0., 1.], [1., 0.]]], requires_grad=True)
    labels = torch.zeros(1, 3, dtype=torch.long)
    valid = torch.tensor([[True, True, False]])
    native = torch.tensor([[0, 1, 0]])
    loss, _ = progress_loss(logits, labels, valid, native, kind="soft_progress")
    loss.backward()
    assert logits.grad[:, 2].abs().sum() == 0
    corrected = logits.detach().clone()
    corrected[:, 1, 0] = 3.
    better, _ = progress_loss(corrected, labels, valid, native, kind="soft_progress")
    assert better < loss.detach()


def test_warm_hints_preserve_known_token_and_unused_masks():
    from radical_decode import warm_start_noise
    embedding = torch.nn.Embedding(10, 4)
    noise = embedding(torch.tensor([[2, 9, 9, 9, 9]]))
    assert torch.equal(warm_start_noise(noise, embedding, [3, 4], 0.), noise)
    blended = warm_start_noise(noise, embedding, [3, 4], .25)
    assert torch.equal(blended[:, :1], noise[:, :1])
    assert torch.equal(blended[:, 3:], noise[:, 3:])
    assert torch.allclose(blended[:, 1:3], .75*noise[:, 1:3]+.25*embedding(torch.tensor([[3, 4]])))


def test_rejected_tail_excludes_corrective_token_and_preserves_alignment():
    from radical_decode import rejected_tail
    proposal = torch.tensor([10, 11, 12, 13, 14])
    posterior = torch.tensor([11, 99, 88, 77, 66])
    # One draft token matches. The next known token is posterior[1] == 99.
    assert rejected_tail(proposal, posterior, 1, "draft") == [13, 14]
    assert rejected_tail(proposal, posterior, 1, "target") == [88, 77, 66]
    assert rejected_tail(proposal, posterior, 4, "target") == []


def test_lazy_verification_preserves_first_rejection_and_corrective_token():
    from radical_decode import lazy_verify_tokens
    proposal = torch.tensor([[0, 1, 2, 3, 4]])
    for mismatch in range(5):
        predictions = torch.tensor([[1, 2, 3, 4, 5]])
        if mismatch < 4:
            predictions[0, mismatch] = 6
        logits = torch.nn.functional.one_hot(predictions, 7).float()
        for chunk in [1, 2, 3, 5]:
            visits = []
            def head(hidden):
                visits.append(hidden.shape[1])
                return hidden
            posterior, accepted = lazy_verify_tokens(proposal, logits[:, :chunk], logits, head, chunk)
            expected = min(mismatch, 4)
            assert accepted.item() == expected
            assert posterior[0, expected] == predictions[0, expected]
            assert posterior.shape[1] == min(5, ((expected//chunk)+1)*chunk)
            assert sum(visits)+min(chunk, 5) == posterior.shape[1]


def test_draft_head_proxy_never_changes_target_verification_head():
    from quantized_draft import DraftHeadTarget
    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 3, bias=False)
        def forward(self, x):
            return self.lm_head(x)
    target = Target()
    draft_head = torch.nn.Linear(2, 3, bias=False)
    proxy = DraftHeadTarget(target, draft_head)
    x = torch.ones(1, 2)
    assert torch.equal(proxy(x), target(x))
    assert torch.equal(proxy.lm_head(x), draft_head(x))
    assert proxy._target.lm_head is target.lm_head
    assert proxy.lm_head is not target.lm_head


def test_shared_compiled_linear_handles_more_than_32_distinct_modules():
    from quantized_draft import attach_shared_linears
    modules = torch.nn.ModuleList([torch.nn.Linear(8, 8).requires_grad_(False) for _ in range(40)])
    x = torch.randn(1, 5, 8)
    expected = [module(x) for module in modules]
    shared = torch.compile(torch.nn.functional.linear, backend="eager", dynamic=True, fullgraph=True)
    names = attach_shared_linears([("test", modules)], shared)
    assert len(names) == 40
    assert all(torch.equal(module(x), reference) for module, reference in zip(modules, expected))


def test_midpoint_straight_through_matches_hard_forward_and_freezes_embeddings():
    from midpoint_conditioning import predicted_embedding
    embedding = torch.nn.Embedding(7, 4).requires_grad_(False)
    logits = torch.randn(2, 2, 7, requires_grad=True)
    hard = predicted_embedding(logits, embedding)
    trained = predicted_embedding(logits, embedding, straight_through=True)
    assert torch.equal(hard, trained)
    trained.square().sum().backward()
    assert logits.grad.abs().sum() > 0
    assert embedding.weight.grad is None


def test_midpoint_only_injects_predicted_slots_and_zero_strength_is_exact():
    from types import SimpleNamespace
    from midpoint_conditioning import MidpointConditioning
    student = SimpleNamespace(layers=[torch.nn.Identity() for _ in range(5)], norm=torch.nn.Identity(), mask_token_id=6, block_size=16)
    target = SimpleNamespace(lm_head=torch.nn.Linear(4, 7, bias=False).requires_grad_(False), model=SimpleNamespace(embed_tokens=torch.nn.Embedding(7, 4).requires_grad_(False)))
    midpoint = MidpointConditioning(student, target, dict(prefix=2, after_layers=2, strength=.5))
    hidden = torch.randn(1, 16, 4)
    changed = student.layers[1](hidden)
    assert torch.equal(changed[:, :1], hidden[:, :1])
    assert torch.equal(changed[:, 3:], hidden[:, 3:])
    assert not torch.equal(changed[:, 1:3], hidden[:, 1:3])
    midpoint.strength = 0.
    assert torch.equal(student.layers[1](hidden), hidden)
    midpoint.handle.remove()


def test_packed_tree_preserves_branch_tokens_depths_and_causal_attention():
    from radical_decode import pack_branches
    proposal = torch.tensor([[0, 1, 2, 3, 4], [0, 9, 2, 3, 4], [0, 1, 8, 3, 4]])
    packed, depths, paths, visible, lengths = pack_branches(proposal, [1, 2])
    assert lengths.tolist() == [5, 5, 5]
    assert packed.shape[1] == 12
    assert torch.equal(packed[0, paths], proposal)
    assert torch.equal(depths[paths], torch.arange(5).expand_as(proposal))
    torch.manual_seed(13)
    embedding = torch.randn(10, 8)
    positional = torch.randn(5, 8)
    hidden = embedding[packed]+positional[depths]
    tree = torch.nn.functional.scaled_dot_product_attention(hidden[:, None], hidden[:, None], hidden[:, None], visible[None, None])
    for i, branch in enumerate(proposal):
        separate = (embedding[branch]+positional)[None, None]
        expected = torch.nn.functional.scaled_dot_product_attention(separate, separate, separate, is_causal=True)
        assert torch.allclose(tree[:, :, paths[i]], expected, atol=1e-6)
        for depth, node in enumerate(paths[i].tolist()):
            assert set(visible[node].nonzero().flatten().tolist()) == set(paths[i, :depth+1].tolist())


def test_tree_cache_compaction_preserves_prefix_and_selected_ancestry():
    from types import SimpleNamespace
    from radical_decode import compact_tree_cache
    class Cache:
        def __init__(self):
            self.layers = [SimpleNamespace(keys=torch.arange(60.).reshape(1, 2, 15, 2), values=-torch.arange(60.).reshape(1, 2, 15, 2))]
        def crop(self, length):
            for layer in self.layers:
                layer.keys = layer.keys[..., :length, :]
                layer.values = layer.values[..., :length, :]
    for path in [torch.tensor([0, 1, 2]), torch.tensor([0, 5, 6]), torch.tensor([0, 1, 9])]:
        cache = Cache()
        prefix = cache.layers[0].keys[..., :4, :].clone()
        expected = cache.layers[0].keys.index_select(-2, torch.cat([torch.arange(4), 4+path]))
        compact_tree_cache(cache, 4, path, main_path=torch.equal(path, torch.arange(3)))
        assert torch.equal(cache.layers[0].keys, expected)
        assert torch.equal(cache.layers[0].values, -expected)
        assert torch.equal(cache.layers[0].keys[..., :4, :], prefix)


def test_leaf_tree_has_one_new_node_per_alternative_and_masks_padding():
    from radical_decode import pack_branches
    proposal = torch.tensor([[0, 1, 2, 3, 4], [0, 9, 2, 3, 4], [0, 8, 2, 3, 4], [0, 1, 2, 7, 4]])
    packed, depths, paths, visible, lengths = pack_branches(proposal, [1, 1, 3], leaves=True)
    assert packed.shape == (1, 8)
    assert lengths.tolist() == [5, 2, 2, 4]
    for i, length in enumerate(lengths.tolist()):
        assert torch.equal(packed[0, paths[i, :length]], proposal[i, :length])
        assert depths[paths[i, length-1]] == length-1
        assert visible[paths[i, length-1]].sum() == length
    # Even if every padded proposal appears to match, it cannot be accepted.
    matches = torch.ones(4, 4, dtype=torch.bool) & (torch.arange(4)[None] < lengths[:, None]-1)
    assert matches.cumprod(1).sum(1).tolist() == [4, 1, 1, 3]


def test_prediction_source_includes_previous_steps_corrective_token():
    from diagnose_tree_precision import prediction_step, first_difference
    assert prediction_step([4, 2, 5], 1) == (0, 0, 0)
    assert prediction_step([4, 2, 5], 4) == (0, 0, 3)
    assert prediction_step([4, 2, 5], 5) == (1, 4, 0)
    assert prediction_step([4, 2, 5], 6) == (1, 4, 1)
    assert prediction_step([4, 2, 5], 7) == (2, 6, 0)
    assert first_difference([1, 2], [1, 3]) == 1
    assert first_difference([1], [1, 2]) == 1
    assert first_difference([1, 2], [1, 2]) is None


def test_ddtree_heap_matches_exhaustive_prefix_mass_and_ancestors():
    import itertools
    from ddtree_baseline import build_ddtree_tree, pack_ddtree
    logits = torch.tensor([[1.7, .2, -.6], [.9, .5, -.8], [.4, .1, -1.2]])
    logp = logits.log_softmax(-1)
    exhaustive = []
    for depth in range(1, 4):
        for path in itertools.product(range(3), repeat=depth):
            exhaustive.append((sum(float(logp[i, token]) for i, token in enumerate(path)), path))
    expected = {path for score, path in sorted(exhaustive, reverse=True)[:10]}
    ids, depths, parents, children, visible = build_ddtree_tree(logits, 10)
    observed = set()
    for node in range(1, len(parents)):
        path, ancestors, index = [], {0}, node
        while index:
            path.append(int(ids[index-1])); ancestors.add(index); index = parents[index]
        observed.add(tuple(path[::-1]))
        assert set(visible[node].nonzero().flatten().tolist()) == ancestors
    assert observed == expected
    packed, depth, paths, mask, lengths, proposal = pack_ddtree(torch.tensor(99), logits, 10)
    assert packed.shape == (1, 11)
    assert torch.equal(proposal, packed[0, paths])
    assert any(sum(token != int(logits[i].argmax()) for i, token in enumerate(path)) >= 2 for path in observed)
    # Every leaf path includes precisely its ancestors; padding cannot be accepted.
    for path, length in zip(paths, lengths):
        path = path[:length]
        assert mask[path[-1]].nonzero().flatten().tolist() == sorted(path.tolist())
        assert depth[path].tolist() == list(range(int(length)))


def test_ddtree_zero_budget_contains_only_verified_bonus_root():
    from ddtree_baseline import pack_ddtree
    packed, depths, paths, visible, lengths, proposal = pack_ddtree(torch.tensor(99), torch.randn(3, 5), 0)
    assert packed.tolist() == [[99]] and paths.tolist() == [[0]]
    assert visible.tolist() == [[True]] and lengths.tolist() == [1]


def test_adaptive_leaves_have_correct_parent_prefix_and_selected_score():
    from radical_decode import pack_adaptive_leaves
    logits=torch.tensor([[2.,1.,0.,-1.],[1.7,1.5,.2,-.9],[3.,.4,.1,-2.],[1.,.9,.7,.4]])
    packed,depths,paths,mask,lengths,proposal=pack_adaptive_leaves(torch.tensor(99),logits,budget=4,topk=3,prefix_weight=1.)
    assert packed.shape==(1,9) and paths.shape==(5,5)
    assert packed[0,:5].tolist()==[99]+logits.argmax(-1).tolist()
    for path,length in zip(paths,lengths):
        path=path[:length]
        assert set(mask[path[-1]].nonzero().flatten().tolist())==set(path.tolist())
        assert depths[path].tolist()==list(range(int(length)))
    # Compute restricted-prefix candidate scores independently.
    probabilities=logits.softmax(-1)
    values=[];mass=1.
    for i in range(4):
        for token in logits[i].topk(3).indices[1:].tolist():
            values.append((mass*float(probabilities[i,token]),i+1,token))
        mass*=float(probabilities[i].max())
    expected={(pos,token) for score,pos,token in sorted(values,reverse=True)[:4]}
    actual=set(zip(depths[5:].tolist(),packed[0,5:].tolist()))
    assert actual==expected


def test_vectorized_ddtree_acceptance_matches_independent_greedy_walk():
    from ddtree_baseline import build_ddtree_tree, pack_ddtree
    generator=torch.Generator().manual_seed(73)
    for _ in range(20):
        logits=torch.randn(4,5,generator=generator)
        packed,depths,paths,mask,lengths,proposal=pack_ddtree(torch.tensor(99),logits,20)
        _,_,parents,children,_=build_ddtree_tree(logits,20)
        for _ in range(10):
            posterior=torch.randint(0,5,(packed.shape[1],),generator=generator)
            path=[0]
            while int(posterior[path[-1]]) in children[path[-1]]:
                path.append(children[path[-1]][int(posterior[path[-1]])])
            gathered=posterior[paths]
            matches=(proposal[:,1:]==gathered[:,:-1]) & (torch.arange(paths.shape[1]-1)[None]<lengths[:,None]-1)
            accepted=matches.cumprod(1).sum(1)
            winner=int(accepted.argmax()); count=int(accepted[winner])+1
            assert paths[winner,:count].tolist()==path
            assert int(gathered[winner,count-1])==int(posterior[path[-1]])


def test_tree_coverage_margin_pushes_missed_target_into_candidate_set():
    from core import tree_coverage_loss
    logits=torch.tensor([[[4.,3.,2.,1.,0.]]],requires_grad=True)
    teacher=torch.tensor([[[0.,0.,0.,0.,9.]]],requires_grad=True)
    loss,metrics=tree_coverage_loss(logits,teacher,topk=2,margin=.2)
    loss.backward()
    assert logits.grad[0,0,4]<0 and logits.grad[0,0,1]>0
    assert teacher.grad is None and metrics['target_topk_coverage']==0
    safe=torch.tensor([[[0.,3.,1.,2.,5.]]],requires_grad=True)
    loss,_=tree_coverage_loss(safe,teacher,topk=2,margin=.2)
    loss.backward()
    assert float(loss.detach())==0 and torch.count_nonzero(safe.grad)==0


def test_reserved_confirmation_rejects_unfrozen_or_changed_decoder(tmp_path):
    import json
    from confirmation_protocol import file_sha, protocol_signature, validate_protocol
    source=tmp_path/'source';source.mkdir();(source/'decoder.py').write_text('pinned')
    manifest=tmp_path/'confirmation.json'
    ids=[str(i) for i in range(128)]
    manifest.write_text(json.dumps({'provenance':{'selection':'confirmation'},'records':[{'problem_id':i} for i in ids]}))
    spec={'eval_manifest':str(manifest),'variants':[{'kind':'ddtree','tree_budget':47}],'output_cap':2048,'repeats':2}
    import pytest
    with pytest.raises(ValueError,match='explicit frozen'):validate_protocol(spec,source,{}, {},'commit')
    spec['phase']='confirmation'
    with pytest.raises(ValueError,match='Freeze selection'):validate_protocol(spec,source,{}, {},'commit')
    freeze=tmp_path/'freeze.json';freeze.write_text(json.dumps({'protocol':protocol_signature(spec,source,{}, {},'commit'),'problem_ids':ids}))
    spec.update(frozen_selection=str(freeze),frozen_selection_sha256=file_sha(freeze))
    assert validate_protocol(spec,source,{}, {},'commit')['phase']=='confirmation'
    (source/'decoder.py').write_text('changed')
    with pytest.raises(ValueError,match='differs'):validate_protocol(spec,source,{}, {},'commit')


def test_history_tree_merges_existing_prefix_without_duplicate_nodes():
    from ddtree_baseline import pack_ddtree
    logits=torch.tensor([[9.,0.,-1.],[8.,0.,-1.],[7.,0.,-1.]])
    root=torch.tensor(99)
    base=pack_ddtree(root,logits,3)
    same=pack_ddtree(root,logits,3,[0,0,0])
    assert all(torch.equal(a,b) for a,b in zip(base,same))
    packed,depths,paths,mask,lengths,proposal=pack_ddtree(root,logits,3,[0,2,1])
    assert packed.shape==(1,6)  # root + primary3 + two new suffix nodes
    found=False
    for path,length in zip(paths,lengths):
        path=path[:length]
        tokens=packed[0,path].tolist()
        found |= tokens==[99,0,2,1]
        assert set(mask[path[-1]].nonzero().flatten().tolist())==set(path.tolist())
        assert depths[path].tolist()==list(range(int(length)))
    assert found


def test_adaptive_budget_truncation_equals_independent_fixed_budget_tree():
    from ddtree_baseline import choose_budget, pack_ddtree
    assert choose_budget([-.1,-.2,-.3,-.4],[1,2,4],.001)==4
    assert choose_budget([-.1,-20.,-30.,-40.],[1,2,4],1.)==1
    logits=torch.randn(4,8,generator=torch.Generator().manual_seed(19))
    policy={'choices':[3,7,15,23],'node_cost':.03}
    adaptive=pack_ddtree(torch.tensor(99),logits,23,budget_policy=policy)
    fixed=pack_ddtree(torch.tensor(99),logits,adaptive[0].shape[1]-1)
    assert all(torch.equal(a,b) for a,b in zip(adaptive,fixed))
