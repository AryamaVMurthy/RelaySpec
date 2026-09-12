"""Ensure an apparently matched argument list cannot hide effective fallback."""
import copy
import pytest
from experiments.handoff_transfer.exact_scope.collect_optimized_methods import validate_runtime


def record():
    return {'contract':{'runtime_profile':'optimized','batch_invariant':False,'request_batch_size':1,
            'runtime_config':{'model':'frozen-target','optimization_level':3,'async_scheduling':True}},
            'effective_runtime':{'compilation_mode':3,'optimization_level':3,'async_scheduling':True,
             'batch_invariant':'0','use_v2_model_runner':True,'quantization':None,'dtype':'torch.bfloat16',
             'tensor_parallel_size':1,'cudagraph_mode':'FULL_AND_PIECEWISE'}}


def test_speculative_config_may_differ_but_runtime_must_match():
    ar=record();sd=copy.deepcopy(ar);sd['contract']['runtime_config']['speculative_config']={'method':'dflash'}
    validate_runtime(sd,ar)
    sd['effective_runtime']['compilation_mode']=0
    with pytest.raises(AssertionError):validate_runtime(sd,ar)


def test_disabled_async_cannot_pass_even_if_both_arms_fall_back():
    ar=record();ar['effective_runtime']['async_scheduling']=False
    with pytest.raises(AssertionError):validate_runtime(ar,copy.deepcopy(ar))


def test_numerical_profile_checks_worker_dispatch_not_just_parent_arguments():
    ar=record()
    ar['contract'].update(runtime_profile='numerics',batch_invariant=True,
                          numerics_profile='invariant-smalltile-rms')
    ar['effective_runtime'].update(batch_invariant='1',custom_ops=['none','+rms_norm'])
    worker={'numerics':{'installed':True,'profile':'invariant-smalltile-rms',
        'rmsnorm_forward_paths':['forward_cuda'],'allow_tf32':False,'bf16_reduced_precision':'False'}}
    ar.update(gpu_before=[copy.deepcopy(worker)],gpu_after=[copy.deepcopy(worker)])
    sd=copy.deepcopy(ar)
    validate_runtime(sd,ar)
    sd['gpu_after'][0]['numerics']['rmsnorm_forward_paths']=['forward_native']
    with pytest.raises(AssertionError):validate_runtime(sd,ar)
