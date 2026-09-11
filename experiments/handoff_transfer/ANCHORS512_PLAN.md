# 512-anchor continuation experiment

Historical precursor plan. The current broader CE/AUF architecture matrix and
active job dependencies are in MATRIX_STUDY_PLAN.md and reports/matrix-jobs.json.
The submission status below describes this earlier stage, not the current queue.

Requested September 12, 2026. Preserve the eight-anchor study as a comparison.

## Same-family primary arms

- Qwen3-4B DFlash proposer retargeted to frozen Qwen3-8B.
- Llama-3.1-8B DFlash proposer retargeted to frozen Llama-3.2-3B.
- ZIP epoch-three initialization, then AUF fusion BA, rank=alpha=56.
- 4,096 distinct records; original frozen-target rollouts, output cap 4,096.
- At most 512 distinct valid anchors per record, sampled without replacement.
  Short records supply fewer. Log actual counts, rather than claiming 512 each.
- Global batch eight: two records/GPU, two GPUs, two accumulation steps.
- 2,000 updates = 16,000 record presentations = 3.90625 epochs.
- Preserve handoff AUF equation and microbatch normalization. Chunk 16 blocks
  for the LM-head objective, with checkpointing and summed numerator/denominator.
  This is numerical batching, not an alteration to AUF support.
- BF16 draft math, FP32 trainable tensors and loss reductions; AdamW,
  learning rate 1e-4, warmup/cosine schedule, zero weight decay, clip norm one.
- Preserve source embeddings, output head, draft backbone and target weights.
- Separate artifacts under handoff-transfer-20260911/anchors512/{q8,llama}.

## Gates and scheduling

Two-update GPU fits test peak memory, finite gradients, changed adapter and
frozen remaining tensors; folded export checked numerically. Four-request,
128-token vLLM comparison must match AR tokens and finish reasons. Gate jobs
31727 (Qwen8, node07) and 31728 (Llama, node06) run serially after 31561.
Cross label job 31734 uses one free GPU released by completed job 31346.
Main pools 31606/31607 wait for both gates and this label pilot; total allocations remain <=4 GPUs.
Full 2,000-step jobs are not yet submitted: measured memory/time gates determine
whether the present batching fits. An OOM requires an explicit, recorded fix,
not silently reducing anchor count.

After successful training: 128 development requests, cap 2,048, natural EOS,
three timing repetitions, matched vLLM settings and full AR token/finish checks.
Compare normal, ZIP, eight-anchor AUF and 512-anchor AUF. Reuse existing matched
baseline measurements explicitly; never count copied measurements as repeats.
Report TPS, ratio to AR and each control, accepted progress, training time,
GPU-hours, actual anchors and peak memory. Update downstream paper assembly
only after verified measurements; no result exists for these new arms yet.

## Qwen-to-Llama extension

Existing pair: Qwen3-4B DFlash proposer -> Llama-3.1-8B target.
The archived tokenizer audit rules out an unchanged target-token AUF port.
The old greedy bridge is usable evidence for design, not a new AUF result.

Required extension: label the target's generated text in the source vocabulary;
align target conditioning to source text prefixes without future-token leakage;
train source-token AUF; retokenize proposals and let the target verify and commit
its own tokens. The AUF formula stays unchanged but its prefix units are source
tokens, so it is not equal to target-token accepted progress. Report separately.

Before GPU training, verify byte-safe prefix alignment, special tokens, causal
feature visibility, source/target position IDs and cache behavior. Inference
must use the same alignment as training. The old singleton token lookup must
not be assumed equivalent to full-text retokenization (the prior audit found
counterexamples). A new 4,096-record cache may be necessary; no direct reuse of
Llama token IDs as Qwen labels is permitted. Begin with exactness and a bounded
training pilot, then expand to the requested budget only if this is valid.
