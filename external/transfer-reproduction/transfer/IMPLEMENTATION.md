# Exact selected computation

## Models and interfaces

Target: frozen Qwen3-8B, hidden width 4,096. Drafter backbone: frozen Qwen3-4B-DFlash-b16, hidden width 2,560. Its original input embeddings and output head come from Qwen3-4B; the 4B base ties these tables. The verifier always uses the full original Qwen3-8B embedding/head. Tokenizer vocabulary and special token IDs are checked during preparation.

There is one trainable object: five bias-free matrices `maps[i].weight`, each `[2560,4096]`, totaling **52,428,800 parameters**. All five initialize with `torch.nn.init.xavier_uniform_`, in index order, after CPU and CUDA seeds are set to 42. The original drafter fusion matrix `F` is `[2560,12800]`; its hidden RMSNorm weight `g` is `[2560]`. Both are frozen buffers.

## Input preparation

Source: `AI-MO/NuminaMath-CoT`, revision in `pins.json`, **only** `data/train-00000-of-00005.parquet`.

Normalize problem text with NFKC, lowercase and collapsed whitespace. Deduplication/template grouping additionally replaces decimal/integer strings with `#`, removes non-word/non-`#` characters, collapses whitespace and SHA-256 hashes the result. Keep the first row per group. This is custom conservative grouping, not semantic equivalence or official upstream curation.

Allocate 20,736 selected examples proportionally to `source`, using floor quotas plus largest fractional remainders. Sort each source by SHA-256(`42:` + group ID), accepting tokenized prompts of at most 1,024 tokens. Sort selected rows by SHA-256(`split42:` + group ID). Partition: first 16,384 train; next 256 reserved development/warmup; remaining 4,096 held-out candidates. Evaluate only the first 128 held-out candidates. Reconstructing the larger selection is necessary to reproduce those 128 IDs.

Training user content:

```
{problem}
Solve the problem and put your final answer within \boxed{}.
```

Evaluation and evaluation-warmup user content:

```
{problem}
Please reason step by step, and put your final answer within \boxed{}.
```

Apply the pinned tokenizer chat template with `add_generation_prompt=True`, `enable_thinking=False`, `tokenize=False`; then encode with `add_special_tokens=False`. The empty template thinking delimiters are part of the saved prompt; no thinking generation is requested. Training solutions are never used.

## Training continuations and features

Generation: frozen 8B target; BF16; seed42; greedy temperature0; response cap4096; max model length5120; max active sequences128; max batched tokens8192; GPU utilization.85; prefix cache off; FlashAttention. Two independent engines queue1024 requests at a time, drawn from eight alternating128-example shards. The original generation default model-generation configuration and compilation settings are preserved in `generate.py` (unlike final evaluation). No mapper participates in generating these data.

For each full prompt-plus-response sequence, independently stratify the prompt and response into consecutive groups of up to four tokens. Within every group choose one position using Python `random.Random`, with a segment-specific seed from the first eight little-endian bytes of SHA-256(`42:{group_id}:{prompt|response}`). Include the final partial stratum. Positions are fixed across both models and all epochs. Token0 and boundary positions follow this same rule; there is no padding or answer-only mask.

Capture auxiliary outputs of zero-based decoder layers **[1,9,17,25,33]** from both targets. The vLLM API registers input-side hooks **(2,10,18,26,34)**, which correspond to those preceding layer outputs. Concatenate in that order: teacher8 width20480, teacher4 width12800. Capture performs complete teacher-forced prefill, not autoregressive regeneration. `max_model_len=5120`, pooling `token_embed`, BF16, max sequences16, max batched tokens8192, unchunked prefill, prefix cache off, GPU utilization.8, FlashAttention. Each `encode` call has at most eight examples. Persist only sampled rows, BF16, with exact positions and identifiers.

Recorded train totals: **21,618,399 full tokens**, **5,416,257 sampled positions**. Paired BF16 payload per position is `2 * 5 * (4096 + 2560) = 66,560 bytes`. Total paired payload360,506,065,920 bytes. Features from both targets are aligned by group ID, position and rollout SHA, never by approximate token matching.

## Loss and optimizer

For sampled token t and layer i, let `x_i ∈ R^4096` and `y_i ∈ R^2560` be the 8B and 4B vectors. Matrices use PyTorch's `linear(x,W) = x Wᵀ` convention:

\[
z_i=W_i x_i,\quad z=[z_1;\dots;z_5],\quad y=[y_1;\dots;y_5].
\]

Define (epsilon exactly10^-6):

\[
N_g(u)=g\odot\frac{u}{\sqrt{\operatorname{mean}(u^2)+10^{-6}}},\qquad
r(a,b)=\frac{\sum_j(a_j-b_j)^2}{\sum_j b_j^2+10^{-6}}.
\]

The per-position loss is:

\[
\ell_t=r(N_g(Fz),N_g(Fy))+\frac15\sum_{i=1}^5 r(z_i,y_i).
\]

Both terms have coefficient1. This is neither cross-entropy nor a draft-token loss. Target/drafter weights receive no gradients. `rms` computes square/mean/reciprocal square root in FP32, casts the normalized vector back to the input dtype, then applies the frozen norm weights. Relative errors are reduced in FP32. Forward matrix operations run under BF16 autocast; mapper master parameters and optimizer state are FP32.

Each example e with n_e sampled positions assigns weight `w_t=1/n_e`. Positions are permuted within each example by a CPU Torch generator seeded42+epoch. Example/shard order remains sorted and fixed. A streaming buffer fills2048-position batches; the final batch can be smaller. With P total sampled positions and N=16384:

\[
L_B=\frac{P}{N\,2048}\sum_{t\in B}w_t\ell_t.
\]

The same scaling is used for the final incomplete batch. Epoch logging reports `(Σ_t w_t ℓ_t)/N`, not the optimizer's scaled batch loss.

Optimizer: fused AdamW; lr10^-3; betas(.9,.999); weight decay0; global gradient clipping1 with nonfinite errors. Exactly3 epochs,2645 steps/epoch,7935 total. Five-percent warmup: `warm=max(1,int(.05*steps))`; for zero-based step s, rate is `(s+1)/warm` before warmup ends, otherwise `.5*(1+cos(pi*(s-warm)/max(1,steps-warm)))`. No validation-loss selection, early stopping or seed search. Save final epoch index2. Epoch-boundary resume includes optimizer, model, step and history.

Observed epoch losses:0.7454407225,0.3302505008,0.2395043797. Recorded fitting wall time887.55s on one L40S. Tensor hashes, rather than just these rounded losses, check exact checkpoint reproduction.

## Deployment folding

Split F into five column blocks F_i, each `[2560,2560]`. Form in FP32:

\[
F'=[F_1W_1\mid F_2W_2\mid\dots\mid F_5W_5]\in\mathbb{R}^{2560\times20480}.
\]

Cast F' to BF16 for deployment. Replace only the drafter's fusion matrix; retain its original hidden RMSNorm. Runtime applies `N_g(F'x)`. The fold removes five separate mapper GEMMs during decoding. Export checks relative MSE below10^-3 against the unfused training computation on actual captured positions; BF16 rounding means algebraic equivalence need not be bitwise equality.

The custom model class uses native vLLM DFlash forward/logits code. Its only runtime integration is to prevent vLLM from replacing the drafter's original 4B embedding/head modules with the 8B target modules. The target itself is unchanged. There is no additional live4B teacher during mapped inference. Tensor storage in this portable export is one safetensors file; the executed deployment used multiple immutable shards with the same values.

## Corrected final evaluation

Native vLLM0.28.0+cu129; BF16; FlashAttention; one request at a time per GPU; TP1; seed0; `generation_config='vllm'`; `max_model_len=4096`; max sequences8; max batched tokens2048; utilization.8; prefix caching off; asynchronous scheduling off. Set `VLLM_BATCH_INVARIANT=1`, `VLLM_USE_V2_MODEL_RUNNER=0`, `OMP_NUM_THREADS=8`.

Disable torch.compile with compilation mode0. Request full CUDA graphs with capture sizes[1,2,4,8,16,32,64]. FlashAttention resolves this to **FULL_DECODE_ONLY**; native decode graph acceleration remains enabled. These settings are essential to this corrected numerical comparison.

Sampling: temperature0; top_p1; top_k-1; min_p0; repetition penalty1; presence/frequency penalties0; seed0; stop token151645; ignore_eosFalse; max new tokens2048. DFlash proposes15 tokens (block16 including the verification bonus). Both speculative pipelines use the same native target verifier.

Timing is `perf_counter` around each synchronous `LLM.generate` call after setup and the four designated warmups. A project-local worker counter tracks vLLM JIT events; compilation-affected requests retry once, with cold time retained. This is end-to-end per-request generation wall time, not the paper's pure decode timer. Speedup is `Σ AR request seconds / Σ tested pipeline request seconds`. Equal output arrays make throughput and wall ratios coincide.

Sources for upstream implementation: [DFlash paper](https://arxiv.org/html/2602.06036v1), [DFlash reference commit](https://github.com/z-lab/dflash/tree/b0375dfb6c9a8820088ade033861d649ce70c8a7), [vLLM release](https://github.com/vllm-project/vllm/releases/tag/v0.28.0). The mapper training objective and data curation are custom. This is not a claim of matching the paper's hardware, backend or timing implementation.
