# Expanded loss × adapter study

## Required comparisons

Original RelaySpec and transfer.zip are initializers/feature-fitting methods,
not CE losses. Keep their unmodified checkpoints as controls, then explicitly
label each token-loss continuation. Frozen original targets throughout.

For each Qwen4 drafter→Qwen8, Llama8 drafter→Llama3, and the separate Qwen4
→Llama8 heterogeneous-tokenizer extension, compare these six architectures:

1. Original normalized RelaySpec interface, initialized from its fitted map.
2. ZIP folded fusion interface with rank56 residual BA (existing handoff arm).
3. ZIP five independently trainable dense layer maps, frozen native fusion.
4. One unrestricted dense target-taps→source-fusion matrix, ZIP initialized.
5. Five independent rank56 residual BA layer maps on frozen ZIP maps, frozen fusion.
6. One unrestricted dense target-taps→source-fusion matrix, fresh initialization
   as a control for benefit from ZIP initialization.

Run uniform CE and AUF for each architecture (12 cells per family, 36 total).
This includes both CE and AUF for the user-requested five-map, final-fusion and
five-BA variants and avoids confounding architecture with loss. Retain normal
RelaySpec and ZIP feature-only controls, AR, and compatible native drafter.
The paper must call added variants CE/AUF continuations, not silently relabel
original feature losses as CE. Include a position-decayed CE reference for the
best CE architecture per family to distinguish uniform CE from DFlash's loss.

## Official reference versus this experiment

Source: https://arxiv.org/html/2602.06036v1#A1 (Appendix A.1), checked 2026-09-12.
DFlash:512 randomly sampled anchors/sequence,6epochs,AdamW lr6e-4,clip1,
cosine4%warmup,maxsequence3072 (4096 Qwen3-Coder),decay gamma7 forblock16,
5 forblock10,4 forblock8. Offline frozen-feature caching is supported.
Our requested experiment:4096 distinct records,4096 output rollout cap,
512 anchor limit,initial matched16000 record presentations (2000updates,
globalbatch8). These are deliberate experimental differences from DFlash.
Report actual anchors for short records; do not duplicate anchors to inflate counts.

## Tuning and evaluation

- All cells: identical training record IDs, rollout files, target checkpoints,
  anchor seed policy, global batch and initial compute budget.
- Initial learning-rate screen:1e-4,3e-4,6e-4; same100updates for every candidate.
  Advance the best two to500updates using the same validation rule.
  Save100/250/500 milestones. Tuning uses a separate fitting-validation split,
  not the final128-request evaluation. Feature loss alone cannot select speed.
- Screen32 validation requests atcap512; verify AR exactness before ranking by
  accepted progress and TPS. Advance the best LR per cell to2000updates.
- On selected checkpoints compare block sizes supported by the frozen drafter;
  record trained block size and any runtime change. Do not assume arbitrary
  block sizes are supported or choose using the final evaluation.
- Final:128 requests,2048-token output cap,naturalEOS,greedy,3timingrepetitions,
  full token+finish equality against matched AR. Report aggregateTPS,ARspeedup,
  ratios against both original andZIP,accepted progress,trainingGPU-hours,
  peak memory and uncertainty. Independent fit seeds for selected conclusions.
- Keep single-request latency comparisons separate from batched serving tests.
  Benchmark selected finalists at concurrency1/4/8 with matched AR; do not
  claim max_num_seqs=8 means the sequential latency workload was batched.

## Implementation and GPU execution

Implement all projections,objective selection,initializer contracts,folding,
runtime input normalization and collectors before broad submission. Unit checks
must establish exact folding, frozen parameters, first-failure support and loss
normalization under chunking. GPU gates check finite updates and deployed AR
exactness. Cross-family needs source-label/target-context alignment and matching
inference; it is not an unchanged shared-vocabulary loss port.

Use BF16,FP32 loss/trainable reductions,cached features,length-aware batches,
checkpointed LM-head chunks and gradient accumulation. Measure before increasing
microbatch; preserve global batch while tuning implementation. Maximum4 GPUs,
prefer2×2-GPU lanes on one node when caches/checkpoints exist there. Do not
migrate active jobs solely to colocate. Routine monitoring interval20–30minutes;
inspect failures or job completion when needed to take the next action.

## Results and paper

Create a machine-readable ledger for every cell,candidate,checkpoint and status;
keep failed and negative outcomes. Only verified128/cap2048 runs enter main
result tables. Add loss×architecture heatmaps,accepted-progress versusTPS,
data/compute curves and total-cost plots as available. Update methods,main
comparisons,contributions and limitations from the measured evidence. No claims
of superiority or cross-vocabulary support without the matching experiment.

Current implementation remains incomplete. Four Qwen512-anchor100-update fits
are verified (fiveBA CE/AUF, dense fusion CE, five dense CE). Cross-family16-record
CE/AUF integration fits are verified; heterogeneous-vocabulary decoding is still
undergoing its GPU gate. Initial family arrays31771/31772 and the two additional
LR arrays31780/31781 are queued. Complete32-request validation arrays31782/31783
follow. Promotion jobs31786/31787 select the best two LRs per cell for500-update
arrays31788/31789. Validation31790/31791 then selects the final LR through
31792/31793, enabling2000-update arrays31794/31795. These jobs are queued
behind verified tuning evidence; none is a completed result. Matched baseline arrays31796/31797 fill missing AR/ZIP repetitions and Qwen
native controls. Final128/cap2048×3 arrays31798/31799 follow verified baselines.
Cross-family full-data work and final paper synthesis remain outstanding. Older baseline pools31606/31607 are held
while the new matrix dependencies are assembled; release them deliberately.
See reports/matrix-jobs.json for repaired failures and current dependencies.

## Cross-family full-data staging (prepared, not launched)

`cross_data_array.sbatch` defines64 chunks of64 distinct records, target rollout
cap4096, with64-request generation batching and4-request feature capture. It
requires both pilot CE/AUF full-token and finish-reason checks to pass before
collecting data. Launch only after checking node06 storage and reserving its
GPU concurrency within the overall four-GPU schedule.

`cross/assemble.py` verifies contiguous offsets, disjoint IDs, shared source
manifest, rollout/alignment hashes and per-chunk causal-capture checks. It writes
a feature index rather than copying the large tensors. `cross/dataset.py` uses
memory mapping, a bounded metadata cache, and feature hashes checked on first
use. Full-data initializer fitting and the full cross trainer still need to be
connected; the16-record integration initializer must not be relabeled4096.

The cross loss supervises source-vocabulary proposals on shared, stable text
prefixes. It is not target-token AUF under a shared vocabulary. Both the train
labels and inference text bridge must retain that distinction in reporting.

`cross/full_train.py` now implements the100/500/2000-update streamed CE/AUF
fits, shared batch8,512-anchor limit, warmup/cosine schedule, clipping, resumable
optimizer state, and frozen-weight/folded-export checks. It rejects the512-record
pilot initializer: a4096-record initializer and index hash are required. This
trainer has syntax validation and data-loader tests but has NOT passed a GPU
full-data run. The paired source-feature cache and full-data initializers remain
the prerequisite implementation work. Chunk-shuffled record order keeps the
metadata cache bounded while visiting every record once per epoch.
