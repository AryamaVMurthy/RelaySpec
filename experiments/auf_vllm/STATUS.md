# AUF study implementation status

Updated 2026-09-11 at approximately 19:39 IST from Slurm and archived artifacts.
Separate branch research/auf-vllm-20260911. Original paper/results unchanged.

## Main finding so far

Qwen4 drafter transferred to Qwen8, 4096 fitting records, 3 epochs, seed42,
selected lr1e-3 for ZIP/CE/AUF. All five-map fits use the original frozen targets.
On the SAME 128 Numina DEVELOPMENT requests, cap2048 with natural EOS, one
warmed timing repetition, vLLM0.28, L40S:

| Method | Tokens/s | Speedup over AR | Exact outputs vs AR |
|---|---:|---:|---:|
| AR | 24.56 | 1.00 | reference |
| Native Q8 DFlash | 168.72 | 6.87 | 128/128 |
| ZIP layer+context feature loss | 173.94 | 7.08 | 128/128 |
| Uniform CE maps | 108.75 | 4.43 | 128/128 |
| AUF maps | 98.81 | 4.02 | 128/128 |

Each produced122768 tokens; 25/128 reached cap. This is not final confirmation
or the four-workload benchmark. ZIP is descriptively3.1% above native; timing
repetitions/seed uncertainty are outstanding. AUF is43.2% below ZIP here.
Source: reports/q8-selected-dev128/aggregate.json and underlying paired rows.

## Completed work

- Exact AUF prefix mask, first-error supervision, detached support, microbatch
  normalization, original frozen teachers, padding/EOS/causal context checks.
- Q8 and Llama complete4096 training data and separate1024 validation records.
- Three selected main fits each for Q8 and Llama: ZIP, CE, AUF;3 epochs seed42.
- Q8 nine-cell512-record LR screen, all selects1e-3; same fixed LR choices
  transferred to Llama/Q14. Initial1e-4 Q8 fits also archived.
- Offline1024-record validation,4096 blocks, epoch3 mean prefix:
  Q8 ZIP5.8091 / CE3.5007 / AUF3.1265;
  Llama ZIP2.6375 / CE2.1504 / AUF1.5315. These are not decoding TPS.
- Rank56 fusion-only LoRA screen,512 continuation records,3epochs:
  ZIP base5.8091 -> CE5.8154 / AUF5.8318 offline prefix. No speedup measured.
- Q8 GPU profiles31354 for AR/native/ZIP/CE/AUF: actual CUDA kernel traces
  verified (31340--39022 kernel events each), CPU events/memory and nvidia-smi
  telemetry captured. Timing runs separate. Kernel/category interpretation
  and hardware-counter analysis are not complete; these are not Nsight runs.
-32k distinct-record manifest prepared with provenance/exclusion checks;
  full32k training remains unrun. Fixed-update resumption passed32-update gate.
- Sixteen implementation tests pass including drafter LoRA merge/freeze test.

## Failures and recovery

- Q14 data31324_1 aborted at vLLM target-capture teardown after512 writes.
  Recovery31444 verified every target SHA/manifest/group/shape/tap/finite tensor,
  completed source capture, verified it, and COMPLETED. All4096 Q14 records
  now prepared. Original failure retained; validation dependency repaired.
- Llama dev31345 failed because vLLM stat logging was disabled; enabled it.
  Retry31443 passed AR then found custom-model module not on worker import path.
  Fixed runtime bootstrap/import identity; retry31445 passed all three4/4 exactness diagnostics and started full AR evaluation.
  No completed128 Llama speed claim yet.

## Live queue at this update (max4 GPUs)

-31445: Llama exactness diagnostics then128x2048 AR/ZIP/CE/AUF, node06 one GPU.
-31327_0: Q14 validation capture, node06 one GPU; part1 follows.
-31447_0 and31447_1: new drafter-LoRA LR screens, node07 two GPUs.
-31447 has12 total cells: draft/fusion x CE/AUF x three LRs,512records3epochs.
-31450: four selected4096-record3epoch continuation fits and three128x2048
  timing measurements each, pending successful31447. Two GPUs maximum.
-31334: three Q14 fits pending validation;31346 Q14dev128 pending fits.

## Direct drafter-LoRA evidence and new comparison

The ORIGINAL paper already has rank32 drafter LoRA versus feature regression,
512 shared records, three matched warm-time budgets, two LoRA update seeds.
Feature145.05/144.80/144.55 TPS vs LoRA119.30--119.52/115.02--115.49/
118.49--119.55 TPS. This is16 development questions, cap256, a previous runtime.
It supports only that bounded comparison, not universal LoRA inferiority.

New main comparison starts from the SAME ZIP4096 checkpoint and reuses the same
4096 unique records. Four continuations: draft attention/MLP LoRA CE/AUF and
fusion-interface-only LoRA CE/AUF. Rank32/alpha32, original target/embeddings/
head/norm frozen. Drafter branch freezes fusion. Three-rate screen per cell:
2e-5,1e-4,1e-3, select smallest within1% of best offline prefix. Both branches
receive equal records/epochs/optimizer batching, not equal parameter counts.
Report initial ZIP costs plus continuation costs;512-record screens also inherit
4096-record ZIP initialization. They are NOT standalone512-data adaptations.
LoRA weights merge into ordinary BF16 drafter weights; full-logit merge check.

## Remaining scope

- Complete Llama/Q14 decoding and new direct drafter-LoRA comparison.
- Four workloads with128 requests each, proper confirmation, principal timing
  repetitions and additional fit seeds; current completed128 is development.
- Full data16--32768, fixed-update and epoch scaling, capacity/MLP/rank,
  regularization/mixed-domain and secondary method experiments as prioritized.
- Profiling analysis, two final standalone Transformers comparisons, all
  evidence-backed figures and separate manuscript. New paper is not complete.

Exact heterogeneous-vocabulary AUF deferred under authorized fallback after
cross-tokenizer alignment/support audit; see CROSS_FAMILY_DECISION.md.
Our reported Qwen3 experiments use a shared tokenizer and token-ID vocabulary;
the mapper adapts hidden representations or the fusion interface, not vocabulary
IDs. Consequently, these results do not demonstrate heterogeneous-vocabulary
DFlash support.

## Follow-up at19:51 IST: workloads and profiling analysis

- Prepared1040 distinct requests:128development+128reserved confirmation+4warmup
  each for MATH, GSM8K, LiveCodeBench and ShareGPT-derived single-turn chat.
  Audited16427 local artifacts and80030 normalized groups, plus complete32k
  calibration and original held-out manifests; no parse failures. Matching is
  normalized exact/template only. Prompt IDs fit both pinned tokenizers.
- Confirmation is unused. Development workload arrays queued:31470Q8,
  31471Llama,31472Q14, each four workloads xthree repeats. AR and every trained
  objective run within each job; Q8 also includes native DFlash. Method order
  rotates over repetitions. Existing Numina results are a separate workload.
- New generic Q8 runner gate31461 follows screens31447; main LoRA31450 now
  follows gate31461. New Q8 workload31470 follows31450. Node07 max2GPU.
  Onnode06 Llama31471 follows31445; Q14chain ends in31472. Each max1GPU,
  for max2GPU/node06 and max4 total. These arrays are queued, not complete.
- Kernel analysis finds96.2--97.0% of recorded kernel duration in matrix
  multiplication. This is a bounded instrumented window; not SM occupancy,
  per-output-token cost, or a substitute for unprofiled timing.
- Added figures/q8_selected_decoding.{pdf,png} with10000 paired request
  bootstrap resamples, and figures/q8_gpu_kernel_share.{pdf,png}. Request
  intervals exclude seed, timing-order and selection uncertainty.

## Follow-up at20:02 IST: seed/scaling scheduling

- Added robustness arrays31480Q8 and31481Llama: six fits each, seeds43/44,
  ZIP/CE/AUF4096records3epochs; one128MATH/2048 evaluation each. Pending main
  workload completion. No additional Q14 fit seeds per agreed protocol.
- Added31478:18fixed-update AUF/CE cells for16--4096records,1024updates,
 32visits/update,4anchors/visit;128-update resumable chunks; full endpoint
  offline validation and128x2048 decoding. Pending31480.
- Added31479: four continuous12epoch fits at512/4096records,AUF/CE, evaluated
  epochs1/3/6/12. Pending31478. This is not yet executed or claimed complete.
- Node07 chains enforce max2GPU; node06 two independent max1GPU chains.
  Higher data counts, ZIP scaling, capacity, mixed domains, final confirmation,
  Transformers checks and paper still remain. Eighteen tests passed previously;
  scaling wrappers pass Python compilation and shell syntax checks, while the
  underlying resumable update trainer already passed its GPU resumption gate.

## Follow-up at20:09 IST: complete LoRA screens and deployment checks

All12cells of31447 completed successfully, approximately5minutes each.
The predeclared smallest-LR-within1% rule selects2e-5 for all four branches:
draftCE/AUF andfusionCE/AUF. Full reports and selections are archived in
reports/draft-lora-screen-31447. Main4096continuations remain pending runtime
gate31461. The gate already verified504627200 projection elements with exact
full-tensor equality for ZIP and checked vLLM's derived fused context-KV buffer.
All later mapped evaluations repeat this check;19 unit tests pass, including
rejection of a stale context-KV buffer. This is additional assurance that the
runtime uses the saved adapter/drafter weights, not a speedup result.
Q14 first512-record validation shard completed; second shard running.

## Runtime gate complete at20:10 IST

31461completed3m42s. ZIP/CE/AUF/native each matched AR4/4 on the new MATH
manifest's diagnostic subset, cap128. Full tensor projection/context-KV checks
passed for all mapped models. Archive:reports/q8-workload-gate-31461.
Main drafter-LoRA fits31450_0(CE) and31450_1(AUF) are now RUNNING on node07,
4096records3epochs at selected2e-5; fusion branches follow within the same
max2GPU array. Llama31445 evaluation andQ14validation31327_1 remain RUNNING
on node06. Total4GPUs. This is diagnostic exactness, not full128-workload data.

## Follow-up at20:27 IST: Llama ZIP result and larger data

Llama8drafter->Llama3target ZIP now completed128 Numina development requests,
cap2048,naturalEOS:186.93TPS vs46.01AR,4.063x,128/128 exact complete outputs,
85900 output tokens. This is one timing repetition and one fitting seed;
CE/AUF evaluations remain inprogress. Replayed locally against raw rows in
reports/llama-selected-dev128. Do not substitute the short pilot numbers.

Preparation31485 completed:16384 rollout manifest and4096 dense records reused
with221327183872 bytes SHA-checked. No GPU used by this preparation job.
Queued31486 captures the remaining12288 target records;31487 adds8192/16384
fixed-update AUF/CE points. Queued31488/31489 generate/capture the added16384
records for32k;31490assembles dense plus original-quarter paired caches;
31491runs32k ZIP/CE/AUF3epoch fits and128x2048 decoding;31492adds the32k
fixed-update AUF/CE points. New32k labels/features/training have NOT run yet.
The active LoRA CE/AUF fits andLlama/Q14 work retain the4GPU cap.
19tests pass; the original quarter-position sampler passed its exhaustive
small-length/boundary check. All16384new prompt IDs match the pinned template.

## Follow-up at 20:39 IST: matched epoch controls and LoRA evaluation

Llama CE completed the same 128 development requests: 142.2308 tokens/s,
3.0916x AR, 128/128 exact outputs, 85,900 output tokens. ZIP remains
186.9255 tokens/s versus AR 46.0053. AUF is still running; no AUF endpoint
claim yet. Raw CE/ZIP/AR rows and local comparison replays are archived.

Both Q8 drafter-LoRA main fits completed (4096 records, three continuation
epochs, rank32, selected LR2e-5). Each trains 9,175,040 parameters with the
ZIP interface frozen. Both epoch3 merged-logit equality checks pass.
Offline prefix lengths are CE5.816895 and AUF5.820312 versus ZIP5.809082;
these are teacher-forced proxies, not measured speedups. Main decoding is
running; no complete timing repetition was available at this check.
Training reports are in reports/draft-lora-main-31450.

New matched-epoch array31494 adds ZIP/CE/AUF three-epoch fits at
16,32,64,128,256,512,1024,2048,8192,16384 records. Existing4096 and queued32768
fits complete this data axis. Each endpoint receives1024-record offline
validation and128-request cap2048 decoding. Array31495 adds ZIP continuous
1/3/6/12-epoch checkpoints at512/4096, matching the AUF/CE trajectories.
These follow31492 with max2 concurrent GPUs. Equal epochs do not imply
equal updates, token exposure, or FLOPs across feature and token objectives.
ZIP fixed-update sampling remains unimplemented and is explicitly rejected
by the wrapper; AUF/CE fixed-update curves must not be presented as such a ZIP
comparison. New wrappers compile and all19 existing tests pass.

Array31496 repeats unchanged ZIP and native controls three times at128x2048,
one sequential request stream per worker, immediately after31450.
Q8 workload array31470 now depends on31496, retaining the max4GPU total.
The initial one-timing ZIP baseline is preliminary until these controls finish.
