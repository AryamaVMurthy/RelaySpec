"""Opt-in, recorded numerical experiments; never changes installed vLLM files."""
import os

_installed=False


def install():
    global _installed
    if _installed:return
    _installed=True
    profile=os.environ['RELAYSPEC_NUMERICS']
    import torch
    if profile=='strict-blas':
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
        os.environ['CUBLASLT_WORKSPACE_SIZE']='1'
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=(False,False)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=(False,False)
        torch.backends.cuda.preferred_blas_library('cublaslt')
    elif profile.startswith('invariant-smalltile'):
        assert os.environ.get('VLLM_BATCH_INVARIANT')=='1'
        from vllm.model_executor.layers import batch_invariant as bi
        old=bi.matmul_persistent

        def smalltile(a,b,bias=None):
            if a.dtype!=torch.bfloat16:return old(a,b,bias)
            assert a.ndim==b.ndim==2 and a.shape[1]==b.shape[0] and a.dtype==b.dtype
            m,k=a.shape;n=b.shape[1]
            c=torch.empty((m,n),device=a.device,dtype=a.dtype)
            sms=bi.num_compute_units(a.device.index)
            grid=lambda meta:(min(sms,bi.triton.cdiv(m,meta['BLOCK_SIZE_M'])*bi.triton.cdiv(n,meta['BLOCK_SIZE_N'])),)
            bi.matmul_kernel_persistent[grid](a,b,c,bias,m,n,k,a.stride(0),a.stride(1),
                b.stride(0),b.stride(1),c.stride(0),c.stride(1),
                NUM_SMS=sms,A_LARGE=a.numel()>2**31,B_LARGE=b.numel()>2**31,C_LARGE=c.numel()>2**31,
                HAS_BIAS=bias is not None,BLOCK_SIZE_M=16,BLOCK_SIZE_N=128,BLOCK_SIZE_K=64,
                GROUP_SIZE_M=8,num_stages=3,num_warps=4)
            return c
        bi.matmul_persistent=smalltile
    else:
        assert profile in ('stock','invariant-o3','invariant-o3-rms','invariant-o3-all'),profile


def kernel_check():
    import torch
    from vllm.model_executor.layers import batch_invariant as bi
    torch.manual_seed(123)
    results=[]
    for k,n in [(4096,4096),(4096,12288),(12288,4096),(4096,151936)]:
        x=torch.randn(129,k,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16).t()
        full=bi.matmul_persistent(x,w)
        for m in (1,8,16,32):
            short=bi.matmul_persistent(x[:m],w)
            assert torch.equal(short,full[:m]),(m,k,n)
        results.append({'k':k,'n':n,'rows':[1,8,16,32,129],'bitwise_equal':True})
        del x,w,full,short
        torch.cuda.empty_cache()
    return results
