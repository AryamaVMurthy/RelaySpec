# Independent audit of the main public-baseline comparison

The strongest supported statement is:

> On the shared 128-question MATH development cohort, the 512-example RelaySpec
> configuration has the highest measured end-to-end token throughput among the
> tested released PARD and frozen-drafter SD² configurations: 193.68 versus 105.63
> and 19.92 tokens/s. Their speedups over their respective runtime-matched AR
> controls are 5.14×, 3.34× and 1.60×. These are measured system configurations,
> not a causal ranking of alignment algorithms.

This statement is stronger and more precise than merely saying that we have
baselines. It does **not** support “beats every competitor”, “best method”, a
quality superiority claim, or a matched-runtime mechanism ranking.

## What was checked directly

I read the generated table, its builder and source registries, the inference
configuration/protocol, and all timed raw records for the three rows. I recomputed
token totals, summed request times and throughput. The following checks passed:

- Each candidate and its own AR have exactly **128 unique request keys**.
- The three candidates have **identical sets of `(problem_id, turn_index,
  repetition)`**, all MATH-500, with identical input-token counts per question.
- Each configuration declares the **2,048-output-token cap**, and every measured
  output has at most 2,048 counted tokens. Generation may stop earlier at EOS.
- All three use **Qwen/Qwen3-8B**, revision
  `b968826d9c46dd6066d109eabc6255188de91218` as the verifier, with separate runtime
  implementations/precision and AR controls.
- Recomputed throughput exactly equals the corresponding registry point values.
  The table's intervals are the existing paired, within-runtime request
  intervals; they have not been replaced by cross-runtime ratios.

Cohort-key SHA-256 (sorted tuple list serialized with compact JSON separators):
`3b6f3b87e52a54e44aae8f69c5d1c6456db7e0b6d4efc5e105d84f86bebf8122`.
Matching input-token counts is an additional consistency check; it is not itself
proof of identical token arrays. The shared manifest, prompt-construction code
and each baseline's audited same-pretokenized-input contract provide the input
provenance.

| Measured row | Counted tokens | Total request seconds | End-to-end tok/s | Own AR tok/s | Own AR speedup [paired 95% CI] | Correct, candidate / own AR |
|---|---:|---:|---:|---:|---:|---:|
| RelaySpec dense, 512 records | 87,529 | 451.920517 | 193.682288 | 37.715676 | 5.135326 [4.903178, 5.374979] | 105 / 104 |
| Released PARD | 85,058 | 805.274837 | 105.626050 | 31.668417 | 3.335375 [3.222296, 3.450610] | 105 / 103 |
| Selected frozen-drafter SD² | 86,925 | 4,364.475479 | 19.916483 | 12.478627 | 1.596048 [1.559597, 1.631882] | 103 / 103 |

AR output totals are 87,737 / 88,374 / 87,152 tokens, respectively. Candidate cap
hits are 9 / 8 / 11; own-AR cap hits are 11 each. Different token totals matter:
throughput ratios are `(sum tokens / sum seconds)_candidate / (... )_AR`, not
just inverse ratios of request times. In particular, PARD's request-time ratio
is 3.4654×, while its token-throughput ratio is 3.3354×.

## Which configurations these rows represent

- **RelaySpec:** `relay_dense_n512`, the seed-1729 dense feature mapper at the
  fixed 8,192-update endpoint, trained on 512 examples. It retargets
  `z-lab/Qwen3-4B-DFlash-b16` to Qwen3-8B, block length 16. The archived benchmark
  source explicitly loads BF16 models with SDPA attention. This is the historical
  small-data/full-answer development campaign, **not** the later FP32 Llama
  evaluation or the BF16/vLLM rollout extension.
- **PARD:** released `amd/PARD-Qwen3-0.6B`, revision
  `f9f650fbab180c26498817718f0db5cae8f25136`, draft length 12, no new-target
  fitting. Transformers 4.51.3, BF16 eager attention, static caches, uncompiled.
  The 4,096-position cache supports the 2,048-token capped evaluation. Inherited
  proposer training is not zero cost and is not included in the “no new fit”
  statement.
- **SD²:** selected KL steering fit at learning rate 4e-6, one epoch over 512
  examples, using frozen `Qwen/Qwen3-0.6B` drafter weights. The measured option
  trains 402,665,472 steering parameters; it is not the fully finetuned SD²
  configuration. Transformers 4.52.4, FP16 target storage, BF16 drafter/steering
  and BF16 autocast. Its one-epoch endpoint was frozen before the full quality
  run after rate and epoch development screens.

The runs use four independent GPU workers, one sequential request per worker,
on L40S hardware. They are separately scheduled campaigns with distinct inherited
drafters, runtime stacks and calibration histories. Timing covers request
prefills and generation/cache work and excludes final detokenization. PARD's
additional verifier-observed replica is **outside** its timed generation; SD²'s
GPU-side verifier observation remains **inside** request time. The appendix
discloses that instrumentation difference. Co-resident controls do not establish
isolated deployment memory.

## Direct ratios: useful only as system-configuration descriptions

Dividing the unrounded raw-record throughput points gives:

- RelaySpec / measured PARD = **1.833660×** (+83.37% token throughput).
- RelaySpec / measured frozen-drafter SD² = **9.724724×** (+872.47%).

Those arithmetic ratios are valid descriptions of these observations but are
**not isolated algorithm gains**. The corresponding ratios of own-AR speedups
are 1.539655× and 3.217527×; normalizing by separate AR runtimes does not remove
all execution, proposer, precision, instrumenting or training confounds either.
No new ratio or cross-system confidence interval has been added to the table.
I recommend leading with the table's absolute measured rates and each system's
own AR speedup, with the qualifier adjacent, rather than pitching “9.7× better
than SD²”.

The measured correctness counts do not support superior answer quality. Full
capped token agreement with each own AR is **36/128 RelaySpec, 28/128 PARD,
42/128 SD²** in this historical BF16/mixed-precision comparison. Do not transfer
the later family extension's 128/128 FP32 results onto these table rows. Actual
verifier decisions and task scoring were audited separately.

## Presentation audit and suggested wording

The current main caption already states that precision, attention and libraries
differ and that each speedup uses own-runtime AR. The appendix supplies the
runtime details and quality uncertainty. This disclosure is correct. The table
header has now been changed to **“End-to-end tok/s”**, retaining **“Own AR tok/s”**
and **“Speedup [95% CI]”**. Only the largest measured throughput point and
own-runtime speedup point are bold; no correctness count or confidence endpoint
is bold. All numerical values and denominators are unchanged.

Suggested compact caption addition:

> Bold denotes the largest measured configuration value. Precision, execution
> and inherited drafters differ; speedups use each system's own AR runtime.

If space permits, “Own-AR speedup [95% CI]” is even clearer as a header. The
existing caption already provides that denominator explicitly, so the requested
shorter “Speedup” header is defensible. “End-to-end” here means timed generation
request including prefill, not model loading, prompt tokenization or output
detokenization.

## Competitors that are not established by this table

- **TriSpec:** no completed, matched 128-question quantitative comparison found
  in the inspected experiment artifacts. It is a related-work/mechanism
  comparison, not a measured inferior method. Prior adapter-only drafter reuse
  must remain credited.
- **PARD²:** not wholly unmeasured. A completed **four-question, 128-token-cap
  pilot** exists at `reports/competitors-length-20260906/pard2-pilot-28196` (and
  its duplicate `run-28196` copy). Its target-dependent and target-independent
  variants record 122.69 and 124.97 tok/s, versus 30.59 own-AR tok/s, with 1/4
  and 2/4 exact sequences. This is **not** a 128-question/2,048-token answer-quality
  comparison. Full campaign configuration files exist but are not completion
  evidence. The PARD row cannot be described as PARD² or as a test of all current
  PARD variants.
- **OmniDraft:** no completed comparable quantitative run found in the inspected
  artifacts. Its related-work differentiation does not justify a speed or
  quality superiority claim.

## Authoritative evidence locations

- `paper/iclr2027/generated/public_baseline_comparison_table.tex`
- `scripts/build_external_baseline_paper_assets.py`
- `reports/mapper-scaling-20260905/small-data-quality-results.json`
- `reports/external-baselines-20260906/pard-quality-full.json`
- `reports/external-baselines-20260906/sd-square-quality-guarded-full.json`
- `configs/submission/baselines/external-verification-protocol.json`
- `configs/submission/baselines/pard-quality-full.json` and `pard-pilot.yaml`
- `configs/submission/baselines/sd-square-compatibility.json`
- Timed raw files under
  `/home/aryamavmurthy/work/RelaySpec/reports/mapper-scaling-20260905/`:
  `small-data-quality-full/run-27806`, `pard-quality-full/run-27916`, and
  `sd-square-quality-guarded-full/run-27934` plus `run-27935`.

The existing external-baseline builder re-audits these source gates and
regenerates the table. This review introduced no GPU runs and edited no main
LaTeX prose.
