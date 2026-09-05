# RelaySpec: next 24 hours on four Turing GPUs

Start: September 5, 2026. Maximum simultaneous allocation: four GPUs.
Observed hardware: node07, four NVIDIA L40S GPUs, 48 GB each.
Budget ceiling: 96 allocated GPU-hours over 24 running hours. Queue delays are
separate. Do not spend the ceiling simply to keep GPUs busy.

## Priority and order

| Order | Question and paper benefit | Quick check, hard limit | Longer work after the check | Estimated four-GPU time |
|---|---|---|---|---|
| 1 | Can completed AR comparisons be reconstructed and scored? This restores the correct primary baseline. | Recover raw files and score saved math outputs on CPU | Recompute throughput, request-time ratios, paired uncertainty and quality from the same records | 0 GPU hours |
| 2 | Where do AR and speculative token choices first differ? This determines the output-preservation claim. | Eight known mismatch prompts per drafter, at most 256 tokens, 10 minutes per job | Replay identical prefixes with controlled precision and call shape. Expand only to unexplained categories | 0.5–2 hours |
| 3 | Does EAGLE-3 retain useful AR acceleration across tasks? DFlash breadth AR runs already exist. | Eight prompts spanning four tasks, at most 64 generated tokens, 10 minutes per job | Complete EAGLE-3 8B and 14B paired AR/source/relay breadth, then score those outputs | 3–5 hours |
| 4 | Are the gains stable across connector fitting seeds? | One short fit and reload, 10 minutes | Reuse seed 1729, fit 1730 and 1731 for each 8B drafter. Evaluate all seeds on the same frozen 128-request set with AR and source controls | 2–4 hours |
| 5 | Which cheaper fitting choices retain AR speed? | Validate saved checkpoint loads and 8-request paired runs, 10 minutes | Reuse existing linear/data/objective/block-size experiments. Add AR only where a matched AR arm is absent. Match data and updates before interpreting either one | 2–4 hours |
| 6 | Is retargeting competitive with inexpensive drafter adaptation? | Implement and check one small update, 10 minutes | One matched-budget representative control after gradients, frozen-weight checks and reload pass | 3–5 hours, conditional on implementation |
| 7 | Can all new evidence regenerate the paper? | CPU checks after every promoted result | Score, aggregate, update figures/tables, compile and inspect the PDF | 0 GPU hours |

Reserve 4–6 wall hours for diagnosed fixes or affected configurations only.
The planned compute is a range, not a promise that every branch will finish in
24 hours. Do correctness and AR evidence first. Defer new architectures,
cross-tokenizer extensions, target drift and serving-engine ports in this window.

## Reuse decisions

- The four main MATH-500 jobs 27357, 27325, 27396 and 27397 are complete.
  Their rank records, configs, manifests and source snapshots were recovered.
  They are not in the rerun queue.
- DFlash breadth jobs 27498 and 27499 are complete and contain AR, source reuse
  and relay for all four tasks. Recover and rescore them.
- Reuse existing selected fitting checkpoints and isolated-memory experiments.
- Reuse block-size 8/16/32 experiments, which already contain AR.
- Reuse the six completed 128-request fitting studies as descriptive comparisons.
  Their data and optimization changes are reported separately. A repeated
  4,096-record manifest is not 8,192 distinct training examples.
- Do not combine AR time from one run with candidate time from another run.

## Promotion and reporting rules

1. A pilot has a Slurm limit of 10 minutes, including loading, and a shorter
   process timeout. It exercises actual CUDA computation, checkpoint loading,
   generation and artifact writing. Instrumented traces are never timing results.
2. A long run starts only after its pilot exits successfully and records have
   complete method/request pairs, positive times, valid token counts and finite
   statistics. Observed output differences are analyzed separately from job health.
3. Every result reports absolute tokens per second and speed relative to AR
   when AR was measured in the same run. Source reuse is secondary. Historical
   source-only ablations retain that denominator explicitly.
4. Mathematical equivalence, measured token agreement and task quality are
   separate claims. A numerical explanation for eight selected examples does
   not explain every mismatch in a 500-request run.
5. Preserve all predeclared workload cells and run outcomes. State narrow
   experimental scope positively without presenting a subset as a full sweep.
6. Use fresh output directories and save code hashes, configs, model revisions,
   hardware, raw outputs and scorer provenance. Never overwrite old evidence.

## Initial execution

- Job 27533: DFlash-8B trace pilot, completed in 1 minute on four L40S GPUs.
- Job 27534: EAGLE-3-8B trace pilot, completed in 59 seconds on four L40S GPUs.
- All four main math runs and both DFlash GSM8K breadth runs rescored locally
  using the existing pinned Qwen math scorer. No regeneration was required.
- Early DFlash traces show changed target logit order or ties at the first
  divergent position, shared by native-target, source-reuse and relay arms.
  Controlled replay is the next diagnostic, not a blanket correctness conclusion.

Detailed job states and findings will be maintained in
`reports/ar-revision-20260905/EXECUTION.md` as jobs are launched and analyzed.

## Updated priority: training records and continuous fitting

After breadth and the numerical diagnostic, prioritize these controls over adding
architectural complexity or another large main-suite rerun.

| Experiment | Fixed quantities | Quantity changed | Pilot | Four-GPU estimate |
|---|---|---|---|---|
| Distinct training records | DFlash-8B map, objective, seed, 1,024 updates, four records/update | Nested sets of 512, 1,024, 2,048 and 4,096 distinct records | One 16-update fit, reload, eight paired requests | 1.5–3 h including four 128-request evaluations |
| Continuous fitting | Same 4,096 records, seed, map and uninterrupted optimizer state | Save at updates 128, 256, 512, 1,024 and 2,048 | Check two intermediate saves and reload within 10 min | 2–3 h including five paired evaluations |
| Seed stability | Selected budget, data and evaluation requests | Seeds 1730 and 1731 alongside reused 1729 | Short fit and reload | 2–4 h for both 8B families |

For each saved map, evaluate AR, source reuse, RelaySpec and native target DFlash
on the same frozen 128-request manifest. Retain the 2,048-token generation cap,
method-order rotation, hardware and timing boundary. Report absolute tokens/s,
AR speedup, native throughput retained, accepted progress, proposed-token
acceptance with an audited denominator, task score, fitting seconds, distinct
records and total record presentations. The main native 8B reference already
exists. Do not run a new native drafter-training job to reproduce its published
training budget in this 24-hour window.

The distinct-data experiment must cycle only its chosen subset when the update
budget exceeds one pass. Save the exact nested IDs. For continuous fitting, add
intermediate checkpoint writing to the existing trainer and preserve optimizer
state throughout one run. Its current final-checkpoint initialization mechanism
alone is not a verified optimizer-resume implementation. Test checkpoint identity
and reload before promotion. Select the deployment checkpoint using development
requests, then evaluate the fixed selected checkpoint on held-out requests.

These two scaling questions replace the generic cheaper-fitting priority in the
original table. Allow roughly 4–6 h for both, rather than adding them on top of
every earlier conditional branch. Within 24 h, the inexpensive drafter-adaptation
control is last and may be deferred. Reserve 4–6 h for fixes, scoring and paper
updates. Queue wait is additional. No long experiment may bypass its successful
10-minute pilot.

Current execution: 27545 is the EAGLE-3 8B breadth run. 27546 follows it for 14B.
The detailed execution record documents completed pilots and their actual gate
outcomes. Check `squeue` before allocating anything else. The four-GPU cap applies
to the sum of all concurrent jobs, not separately to each experiment.
