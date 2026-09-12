# Locked experiment configuration — batch8 per GPU

Selected node: Turing node07, four L40S GPUs. Each token fit uses exactly one GPU
and batch8, without gradient accumulation or cross-GPU gradient synchronization.
Up to four different jobs run concurrently. This supersedes the two-GPU lanes.
No further training batch-size tuning. Batched warmup is used in subsequent vLLM jobs; warmup and any compilation-affected first pass remain outside timing. Dataset and evaluation selections stay fixed.

## Transfers

| Family | Source/drafter interface | Frozen new target |
|---|---|---|
| q8 | Qwen3-4B | Qwen3-8B |
| llama | Llama3.1-8B | Llama3.2-3B |
| cross | Qwen3-4B | Llama3.1-8B |

Qwen uses a shared tokenizer/token-ID vocabulary. Cross-family runs use the explicit
text/tokenizer alignment bridge; they are not evidence of direct vocabulary-ID reuse.
All target transformers and all drafter parameters outside the named interfaces remain
frozen. There is no target LoRA. Existing Qwen/Llama initializer checkpoints are reused;
new cross-family original/ZIP initializers are fit for one epoch each.

## Fits per family (revised September 12)

| Cell | Trainable interface | Objective |
|---|---|---|
| normal_ce/ce | Original RelaySpec normalized linear interface | Token CE |
| dense_fusion/ce | ZIP initializer folded into one fusion matrix | Token CE |
| five_maps/ce | Five dense matrices, frozen native fusion | Token CE |
| five_maps/auf | Five dense matrices, frozen native fusion | AUF |
| dense_fusion/auf | One full fusion matrix | AUF |
| five_ba56/ce | Five BA updates, rank56, alpha56 | Token CE |
| five_ba56/auf | Five BA updates, rank56, alpha56 | AUF |
| feature_ce (completed Qwen/Llama only) | Original normalized linear interface | Soft-target feature CE |
| forward_kl (completed Qwen/Llama only) | Original normalized linear interface | KL(source || mapped) |
| reverse_kl (completed Qwen/Llama only) | Original normalized linear interface | KL(mapped || source) |

ZIP+CE and one-fusion-matrix+CE are one cell, not duplicate experiments.
Twenty-one retained token fits: seven per family. The six already-completed Qwen/Llama feature fits are historical artifacts only; cross-family feature CE/forward KL/reverse KL are cancelled, and no further feature-loss fits or dedicated timing measurements are authorized. Baseline evaluations per family: AR,
unchanged original RelaySpec feature-MSE initializer, unchanged ZIP initializer.
AUF supervises the correct prefix plus its first failure. No exponential positional
weighting or additional MSE/KL term is mixed into token CE/AUF. Feature CE/KL uses
softmax over normalized fused feature coordinates at temperature1; no extra positional
weights. Feature CE and forward KL have identical gradients with a fixed teacher.

## Token fitting

- 4096 distinct training records; rollout cap4096 generated tokens.
- One epoch: 512 updates ×8 records =4096 record presentations.
- Up to32 distinct eligible anchors per record, sampled in the response; short records
  provide fewer. Clean anchor and padding excluded from prediction loss.
- Per GPU batch8; one GPU per fit; accumulation1. CE/AUF normalized within this batch8.
- BF16 operations, FP32 trainable interface weights/loss reductions, cached frozen features.
- Fused AdamW, weight decay0, gradient clip1, 5% warmup then cosine decay; seed42.
- Loss-processing chunks32 blocks; native checkpoint block sizes retained.
- Final checkpoint only for main comparisons. A two-update check verifies fit/export
  integrity before each full same-family fit; full128 evaluation verifies deployment.
- Different batching changes random anchor draws and floating-point reductions relative
  to the archived two-GPU pilot; all new comparison cells use the same batch8 protocol.

| Interface | Qwen4→8 LR | Llama8→3 LR | Qwen4→Llama8 LR |
|---|---:|---:|---:|
| Original / five dense maps | 1e-4 | 1e-4 | 1e-4 |
| Dense fusion | 1e-4 | 6e-4 | 1e-4 |
| Five BA56 | 3e-4 | 6e-4 | 1e-4 |

Rates reuse existing validation selection; cross-family rates are untuned. No fresh LR,
seed, capacity, data-size or regularization sweep. Feature-only fits use one cache epoch,
position batch2048 and LR1e-3; these bypass block anchors and are not512-update token fits.

## Evaluation

- vLLM0.28 primary backend; same fixed128 requests per family, cap2048 output tokens.
- Greedy temperature0, natural EOS, seed0; cap is a maximum, not forced length.
- Submit128 requests together; max_num_seqs128, max_num_batched_tokens8192,
  max_model_len5120, GPU memory utilization0.8, prefix cache disabled.
- vLLM schedules active requests within KV-cache capacity; submission batch128 does not
  assert all128 remain resident simultaneously. This is a target setting, not a measured
  maximum until the full run succeeds. Only explicit memory failures trigger128→64→32→16→8.
- Lock the passing serving batch for AR, original, ZIP and all methods within a family.
- Three timing repetitions, one training seed. Setup/compilation/warmup excluded; logged.
- Report aggregate generated tokens / batch wall time, speedup against matched AR and
  original/ZIP, acceptance/progress, full output-token equality and finish equality.
- GPU utilization, memory, power and temperature recorded. Batched throughput is distinct
  from the older single-request latency numbers.
- Final Transformers checks remain limited to1–2 representative completed cases.

## Initial parallel launch

Jobs32239–32242 on node07: Qwen five_maps/CE, five_maps/AUF, dense_fusion/CE,
and five_ba56/AUF respectively. Each requests one GPU and executes batch8.
Llama token/paired caches are being staged from node06, which was fully allocated.
All remaining fits, reference evaluations and cross-family data preparation are
submitted under the same total four-GPU ceiling. Each family first completes its shared baselines and serving-batch selection;
remaining evaluations then run independently, reading the completed reference
artifacts. Reference comparison JSON writes are atomic. Superseded job ledgers/results are retained separately.

## Execution evidence
Initial four batch8 fits completed with export verification in285.22–292.44 fitting
seconds each. Qwen batch128 AR repetition0 completed128 requests,122768 generated
tokens,99.8542 measured seconds,1229.4725 aggregate tokens/s. No trained-method
speedup claim follows until its matching full output comparison passes.
Transformers jobs32304–32306 run one128-request/2048-cap pass each for AR and Qwen
five-map CE/AUF after the entire vLLM matrix. The standalone decoder currently
supports one request at a time; these are backend confirmation checks with separately
reported timing, not substitutes for the batch128 primary results.

After the final audit passes, its CPU job archives compact deployment interfaces,
unmerged token-fit parameters, model/config provenance, calibration manifests,
evaluation token outputs and available GPU telemetry under the control directory's
`outputs/exact32e1b8/artifacts`. The archive has a SHA256 manifest; large frozen
model weights and feature caches remain referenced rather than duplicated.
The collector also produces CSV/Markdown tables from complete audited timing
triplets, preserving benchmark job IDs and physical GPU UUIDs in the audit.
It checks exact group-ID disjointness between each family's fitting records,
evaluation requests and warmup requests, using the assembled index for cross-family
data. This is an exact-ID check, not a semantic near-duplicate assessment.

## Physical-GPU timing correction (September 12)

Qwen feature CE and forward KL exported byte-identical weights and produced identical
acceptance counters, but their aggregate throughput differed across physical L40S
cards. Raw cross-card relative speedups are provisional. The final collector requires
AR, original RelaySpec, ZIP and each method to share a physical GPU UUID, evaluation
manifest, serving batch and repetition index. Matching a GPU does not eliminate
variation over time; three timing repetitions and the one-fitting-seed caveat remain.

Job32325 supplies missing references with four independent one-GPU workers after the
main vLLM matrix; existing references are reused only on their recorded physical GPU.
Workers skip family/card combinations absent from all seven retained token-method evaluations.
The reference plan refuses to run from an incomplete family. This adds at most81
reference passes to the revised108 vLLM passes (up to189 total, including six historical feature evaluations),
without adding training variants or changing the21 retained token fits. Transformers
jobs32304–32306 depend on32325; final audit32307 follows those checks. The collector
writes separate provisional and GPU-matched tables and rejects incomplete matched
comparisons at finalization.

Cross capture array32262 now permits four independent chunks (Nice100, lower priority
than ready evaluations). It continues to share node07 with evaluation jobs and fills
all four GPUs once those finish; every study job is pinned to node07, which has four
physical GPUs. This avoids leaving two GPUs idle under the original array throttle2.
Record counts, chunk contents and fitting/evaluation configurations are unchanged.

## Native tokenizer normalization repair

Cross chunk12 exposed a valid rollout whose prompt contains U+2001 spaces. Qwen's
native NFC tokenizer maps them to U+2003, whereas Llama preserves the original
text. The former raw-text round-trip guard rejected all597 otherwise valid source
prefixes. Alignment now compares decoded source text with the source tokenizer's
native normalized prefix, while retaining exact full-sequence source-prefix token
ID equality and strictly past target conditioning. It does not normalize or replace
the target rollout, change training records, or alter the inference bridge.

The first1024 records were audited; chunks7 and12 contain affected text. Repair
array32332 rebuilds these two alignment/paired-feature chunks, preserving prior
artifacts in their repair-history directories. Assembly32263 explicitly depends on
the remaining original chunks and this repair array. Chunks14 and15 began from old snapshots; both were audited and contain no
normalization-affected text. Later chunks use the corrected source snapshot.

## CPU alignment batching

Subsequent cross-data chunks align up to four records concurrently on allocated CPU
cores (`min(4, SLURM_CPUS_PER_TASK)` workers). Spawned workers load tokenizers only;
ordered collection retains the exact row order and per-record algorithm. The serial
and parallel outputs were byte-identical on actual short, full4096-token and Unicode
regression rollouts; evidence is in `alignment-parallel-audit.json`. This changes
preprocessing scheduling only, with the existing four-GPU ceiling, training batch8,
and evaluation batch128 intact. Each chunk records its actual `alignment_workers`.

The paired-feature capture stage now reuses generated-token anchors from the
hash-verified `source-label-alignment.json`. It checks selected prompt anchors
separately, verifies source-sequence identity, and retains the identical source
IDs, target/source gather positions, labels and equal-record weights. Reuse requires
record identity, rollout/label hashes and native-normalizer provenance. Actual-tokenizer
comparisons covering short, full-length and Unicode regression records passed;
`paired-alignment-cache-audit.json` records the equivalence check. This avoids a
second scan of already-verified generated prefixes without changing the sample.

## User scope reduction — September 12

Cancelled pending fit jobs32272–32274 and evaluation jobs32301–32303 before execution.
The active cross-family variants are original-interface tokenCE, dense-fusion CE/AUF,
five-matrix CE/AUF and five-BA56 CE/AUF. Job32325 now depends only on
cross evaluations32294–32300. Completed Qwen/Llama feature-loss artifacts are retained
in the27-fit archive/audit, but excluded from further dedicated physical-GPU reference
measurements and the21-cell matched primary comparison requirement. No new feature
CE, forward-KL or reverse-KL training/evaluation is scheduled by the launcher.
All data sizes, epoch counts, anchors, token losses and batching remain unchanged.

## Incomplete UTF-8 tail repair — September 12

Chunk47 row3 reaches its4096 generated-token cap inside the final square-root
character. Its last target token contains only the first two UTF-8 bytes of that
character. A strict byte-level check now distinguishes this incomplete suffix
from malformed interior bytes and literal replacement characters. All4194 target
IDs remain unchanged. Source alignment uses the last complete target-token/text
prefix (4193 tokens); the affected record retains3805 eligible anchors and972
paired fitting positions. `unicode-tail-audit.json` verifies the actual rollout,
complete-prefix label equivalence and paired-cache equivalence. Every affected
alignment summary records its excluded trailing token count. Other records retain
the original algorithm. Repair32446_47 reruns the failed chunk from its existing
rollout; assembly32263 depends on this repair and original chunks48–63.

## Overlap initializer and token fitting

After full4096 assembly passed, ZIP fitting completed one epoch (762 updates,
1559660 paired positions,201.52 fitting seconds). `cross-zip-readiness.json`
verifies the export, checkpoint and index hashes. Jobs32265,32266,32268–32271
were released from the combined initializer dependency at that point, so three
ZIP-dependent token fits can overlap the remaining original-interface fit on
node07's four GPUs. Original-interface tokenCE32267 still waits for32264.
First cross evaluation32294 now explicitly requires both32265 and32264; all
subsequent evaluations continue to require its completed shared baselines.
This changes scheduling only, with no new fitting variants or training settings.

## Cross-family evaluation runtime repair — September 12

First evaluation32294 completed AR repetition0, then failed during the original
interface pass when a mixed prefill/decode batch accessed request slot64. The
installed vLLM stores next_prefill_tokens as [lookahead, request slot], while
the bridge indexed it as one-dimensional. The bridge now changes only the
first-lookahead anchor row, preserving the native buffer shape, other rows and
the original request-state tensors. Six bridge/runtime tests pass, including
a regression that failed before the fix at sparse slot64. Replacement32457
runs the same128-request,2048-token, batch128 evaluation, reusing valid completed
AR output through the existing contract check. The failed job and logs remain
recorded; downstream comparisons now wait for the replacement. No fitting
objectives, training settings or experiment variants were added. Feature CE,
forward KL and reverse KL remain disabled for all future runs.

## Overlap remaining full adapter measurements

After original and ZIP repetition0 both passed128/128 output and finish checks
at batch128, array32458 (tasks1–6, at most three concurrent GPUs) was released
to measure the six existing pending adapter cells while32457 finishes shared
references and five-matrix CE. Each array task runs the same three full128/cap2048
measurements into the existing exact evaluator's output paths using its benchmark
module and argument contract. It checks output agreement against the completed
AR repetition0, but does not declare final evaluation complete. Jobs32295–32300
wait for both32457 and32458, reuse these cached measurements, and perform the
original matching-repetition reference checks. The final collector independently
recomputes all comparisons. This adds no experimental cells, fitting, seeds,
timing repetitions or probes. All jobs remain pinned to node07's four GPUs.

## Archive preflight and executed source preservation

A read-only preflight verified all388 expected training/calibration files,
including3.168GiB of unmerged trainable parameters. All21 token-fit source
snapshots and the completed feature/evaluation snapshots are present. The final
archive now also preserves compressed executed source snapshots for training,
evaluation and Transformers confirmation, plus the overlay/vendor support code.
It excludes model weights, data, caches and environments from source bundles.
This retains the cross-family runtime correction with the resulting artifacts;
archive tests verify code contents and reject missing snapshots.

## Completed primary matrix — September 12

All21 retained token fits and all21 primary evaluations now pass the independent
audit. The six historical feature-loss cases remain complete but withdrawn from
further work, giving27 archived fit/evaluation cells. Cross-family outputs and
finish reasons match AR on all128 requests for each of three timing repetitions
in every primary cell. Parallel measurement array32458 and final validation
jobs32295–32300 completed without adding or repeating experimental cells.

Physical-GPU reference job32325 is running next. Only six primary comparisons
have complete matched-card triplets at this audit checkpoint; the other ratios
remain provisional. Transformers confirmations32304–32306 and final archive32307
remain pending. The new-paper plan has not been issued or launched.
