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
