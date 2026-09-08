# Transfer figures and diagnostic findings

Built by `scripts/build_transfer_visuals.py --root . --output paper/iclr2027`.
The importable API is `build(root: Path, output: Path)`; add `scripts` to the
import path, as for the other paper builders.

The builder re-runs `build_family_extension_paper_assets.build` in a temporary
directory, compares its complete result with the current source evidence, and
checks all raw token arrays again before plotting. It changes no LaTeX, shared
builder or existing family evidence file. No new GPU work was run.

## Files and placement

- **Main:** `figures/visual_transfer_main.pdf` (6.2 × 2.65 in). Two independent
  reference-normalized panels: family portability versus matched FP32 AR, and
  rollout-trained reuse versus matched BF16/vLLM native DFlash. The chart names
  both datasets/runtimes and annotates exact token agreement.
- **Appendix:** `figures/visual_transfer_request_gain.pdf` (6.2 × 2.75 in).
  All 128 paired request-speed ratios per family, versus generated target-token
  length. Fixed bins [0,256), [256,512), [512,1024), [1024,2049) show aggregate
  request throughput ratios and paired bootstrap intervals at their median
  observed length. No fitted trend or significance-selected subgroup is used.
- **Appendix:** `figures/visual_transfer_verification.pdf` (6.2 × 2.8 in).
  Effective output target tokens per verification-loop call and the paired
  association between progress changes and request-speed changes.
- **Appendix:** `figures/visual_transfer_lengths.pdf` (6.2 × 2.55 in).
  Output-length and prompt-length empirical CDFs, with output-cap incidence.
  These are observed capped lengths, not uncensored generation-length estimates.

Each PDF has a same-name 200-dpi PNG. All plotted numeric values, per-request
points, uncertainty definitions and input SHA-256 hashes are saved in
`generated/transfer_visuals_evidence.json` (308 inputs). PDF creation/modification
timestamps are removed; the builder verifies that visible text stays inside the
fixed canvas. A repeated build into a fresh temporary directory matched all PDF,
PNG and evidence hashes byte for byte. The PNGs were inspected at publication
aspect ratios for clipping, label overlap, and legibility.

## New descriptive insight supported by these plots

The selected Qwen→Llama fit is faster on **122/128** individual requests. Effective
output target tokens per verification call increase from **3.3075 to 3.7820**;
total calls fall from **28,000 to 24,487 (−12.55%)** for identical output sequences.
The gain appears across all four fixed output-length bins (aggregate speed
ratios 1.131, 1.149, 1.122 and 1.151). This links the aggregate speed improvement
to reduced verification work, without claiming a causal data-count ablation.

The larger Llama→Llama fit is faster on **58/128** individual requests. Calls are
**16,066 versus 16,094 (+0.17%)**, and effective tokens per call are **3.8969 versus
3.8901**. The absence of a resolved aggregate gain also appears in verification
efficiency. This negative result is retained rather than hidden by a main-panel
speedup relative to AR.

## Suggested captions

**Main:** “Frozen-drafter reuse across complementary settings. (a) Request
throughput relative to matched FP32 AR on 128 MATH-500 requests per model pair:
historical 4,096-record fits and selected larger fits. (b) The independent
16,384-rollout Qwen experiment relative to native DFlash, on 128 Numina requests
in BF16 vLLM, with the GPU assignment reversed in the repeat. Both use a
2,048-token output cap and preserve all 128 reference token sequences. Horizontal
bars are paired 95% request-bootstrap intervals, conditional on checkpoints and
runtime; they exclude fitting-seed, selection and repeat-to-repeat uncertainty.
The panels have different references, cohorts and runtimes.”

**Request gain:** “The cross-family gain is distributed across requests and
observed output lengths. Each point is old/selected request latency for identical
output tokens; diamonds aggregate the four fixed length bins with paired 95%
request-bootstrap intervals. The bins contain 49/42/26/11 Llama requests and
31/45/25/27 Qwen→Llama requests. The common 2,048-token cap censors longer outputs;
this is descriptive analysis of the frozen 128-request evaluation.”

**Verification:** “Fewer verification calls explain the direction of the
cross-family speed improvement. (a) Total final output target tokens divided by
total verification-loop forwards, with 95% request-bootstrap intervals; prefill
forwards are excluded, while the final output count includes the first token and
is EOS/cap-trimmed. This is effective output progress, not proposal acceptance
fraction. (b) Each point pairs selected/baseline progress and request throughput
for one identical output sequence; the dotted diagonal indicates equal ratios.
The selected cross-family fit reduces total verification calls by 12.5%; the
Llama fit does not.”

**Lengths:** “The 128-request evaluation includes diverse observed lengths.
The same question cohort is evaluated for both pairs using the target tokenizer.
Output lengths are capped at 2,048 tokens: 8/128 Llama and 24/128 Qwen→Llama
requests hit the cap. These capped empirical CDFs do not establish answer quality
or the distribution of unrestricted completion lengths.”

## Scope notes

Baseline and selected fits differ in recipe as well as data count. The Llama
selected checkpoint is epoch 6 of a 12-epoch schedule on 16,384 records; the
cross-family selected checkpoint is epoch 3 of a 24-epoch schedule on 8,192
records. The richer-rollout experiment is a separate method/runtime/data recipe.
Confidence intervals are conditional paired-request resamples, not repeated-run
or fitting-seed uncertainty. Exact matches compare generated target token arrays
against the runtime-matched AR reference and do not measure answer correctness.

Exploratory family-scale fitting/block plots already exist in
`reports/family-scale-20260907/figures/`; they were not duplicated because their
two-request screening and eight-request tuning cohorts add selection evidence,
not a second confirmation of the 128-request findings.
