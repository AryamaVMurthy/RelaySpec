# External baseline completion after the small-data budget study

Large-data scaling remains paused. All further fit pools are512 or2048
distinct records. Keep the historical larger-data evidence and cache.
The full paper objective, including transfer, final confirmation and
bounded autoresearch after fixed baselines, remains active.

## Verified current evidence

- SD-square CPU setup27867 passed in1m20s with two CPUs and no GPU.
  Public source00f578aa and Transformers4.52.4 import under Python3.12.
  Its README specifies3.12 despite the pyproject declaring3.13.
  The native Qwen3-0.6B checkpoint is pinned and SHA-verified. The released
  steered checkpoint includes drafter tuning and is not relabeled frozen.
- Four-GPU pilot27869 finished in1m17s. Both public TVD and KL objectives
  trained only the402,665,472 steering parameters. Four updates visited
  four records from the declared512-example pool, not the whole pool.
  Training took1.24--1.35s. Duplicate weights, losses and gradients match
  exactly. Zero-guidance identity, frozen-drafter hashes, frozen-parameter
  version/storage checks and trainable/optimizer reload passed.
- The original exact-AR gate FAILED: both replicas of the second prompt
  first differ at output49. This remains a failed gate.
- Numerical replay27871 passed in1m01s. Same effective98-token prefix
  and logical position, with physical cache prefix129. Under SDPA, AR's
  logits favor59 over87 by36.25 versus36.0, while SD-square reverses that
  ordering. Eager attention reverses the choices again. Eager is not a
  solution to exact identity. The public AR helper also differs from
  correct cached AR on the measured first16-token output.
- Causal replay27877 passed in1m00s. Perturbing the six future query
  tokens preserves the first three query logits bit-for-bit under both
  SDPA and eager. Incoming cache copies are exact and independent, the
  original cache stays unchanged, and baseline replay logits match the
  actual verifier. This excludes future-query leakage in the tested call.
  It does not identify an exact CUDA kernel or prove a global guarantee.
- PARD's earlier cache/query replay27840 already establishes an analogous
  numerical sensitivity and causal intervention. Its exact-AR gate stays
  failed. PARD needs no new-target fitting.

Audited registries live in reports/external-baselines-20260906. Raw JSON
is collected under the original RelaySpec repository. Large checkpoint
and optimizer files remain on compute scratch. No long run is justified
merely by these compatibility results.

## Prospective checks and comparisons

The source-bound declaration is
configs/submission/baselines/external-verification-protocol.json.
It keeps every old exact-AR failure and separates three requirements:
faithful implementation, actual verifier decisions, and task quality.
Every committed token must equal the actual greedy verifier. Preserve
EOS and cap overshoot in raw records and charge complete request time.
Do not use the public SD-square AR helper as the reference. Use correct
cached AR with the same target weights, dtype, attention and token IDs.
Measure exact sequence agreement descriptively. Do not infer accuracy
from that agreement or describe a numerical mismatch as a quality pass.

1. Run the complete512-example SD-square fitting path inside a ten-minute
   pilot: batch4,128updates, learning rate1e-5 to1e-6, warmup8, two exact
   replicas each of TVD and KL. Exercise collation/loss masking, actual
   pool coverage, frozen weights, optimizer export/reload and decoding.
   Retain a separate compatibility result if numerical identity differs.
   A prospective pilot must pass its declared checks before any longer run.
2. If that pilot passes, compare4e-6 and2e-5 for each objective in a
   second four-GPU batch. Together with the pilot this gives six unique
   objective/rate settings. Account for all search cost. Keep public
   steering architecture and source statements unchanged.
3. Decode all six settings plus the initial unsteered drafter and
   runtime-local AR on the same16 exposed requests,256-token cap. Freeze
   rate/objective choice before larger quality data. If more training is
   warranted, use explicit warm-budget trajectories and preserved
   optimizer state, not a new large-data sweep.
4. Run PARD's prospective16-request development comparison with its
   pinned public runtime and matched correct AR. Its earlier exact-AR
   failure remains in the record. Keep inherited training separate from
   zero new-target fitting.
5. For selected external baselines, first run8 requests with2048-token
   caps, then the required128-request exposed quality comparison after
   the short path passes. Report quality intervals, cap counts, exact
   agreement and full request timing. No cross-runtime algorithm-only
   ranking: SD-square's public mixed precision, PARD's eager runtime and
   RelaySpec's runtime are different system configurations.
6. Only after fixed baseline and transfer evidence is complete, start
   the bounded autoresearch phase. Freeze final candidate/checkpoint,
   model/scorer revisions and comparison before untouched confirmation.
   Keep the existing one-percentage-point quality margin and95percent
   throughput-retention criterion. Inconclusive evidence stays inconclusive.

The work above completes a missing baseline track, not the entire paper.
The remaining small-data transfer/family, composition/complexity,
mechanism/serving, anonymous reproduction and final visual-review work
must still be checked against the original execution plan.

## Current execution checkpoint

Full-pool pilot27888 and rate screen27890 passed in2m04s each. Every fit
visited512 examples in128 batch4 updates. Training took47--48seconds.
The first common decoding job27895 was cancelled at3m47s because its
drafter and steering retained FP32 training storage. Preserve partial
artifacts and exclude its timings. The corrected campaign follows public
eval.py with BF16 drafter/steering and FP16 inherited target, under BF16
autocast. Verify original FP32 training fingerprints before conversion
and separately fingerprint the actual BF16 inference tensors.

Corrected inference job27898 was cancelled at5m13s after measured
SD-square requests took10--15seconds for256 tokens. The full nine-method
sixteen-request batch would exceed the pilot limit once setup and
verification are included. Preserve partial artifacts without a completion
claim. Execute the same fixed comparison in two disjoint eight-request
shards, each with four GPUs and a ten-minute allocation. Aggregate only
after both source-bound gates pass and the sixteen prompts are disjoint.

Launched SD-square shards27900/27901 from3e96e78. The second depends on
the first passing. PARD campaign27903 frome3bb73e depends on the second
shard and uses the same sixteen development prompts. All jobs request
four GPUs with ten-minute limits. PARD verifies a separately observed
replica of every timed request against its actual target argmax decisions
and checks raw tokens and acceptance against the trace-free timed run.
Public generation statements remain unchanged, tested by AST equality.
Auditors rebuild source-bound gates before creating result registries.

SD-square shards27900/27901 both passed in9m07s/9m00s. The source-bound
audit verifies all144 outputs and identical inference fingerprints across
sixteen disjoint prompts. The best fitted setting is KL at4e-6,18.6639
tokens/s versus18.8321 for the independent drafter. Its independent-ratio
interval is0.96192--1.01624, so this screen establishes no steering gain.
Freeze that objective/rate before the1/2/4/8-epoch512-example screen.
Every initial fit uses95,835 supervised non-padding positions capped
at192 tokens per record. Report this separately from feature regression's
supervision and retain all six rates and duplicate fitting costs.

PARD27903 passed in1m22s. All32 measured rows and sixteen verification
replicas reproduce, with strict actual-verifier checks. PARD reaches
98.5975tokens/s against31.2725 for its eager AR control, ratio3.15285
[2.83051,3.55756]. Four of sixteen capped sequences exactly match AR.
Keep this public-runtime development result separate from task quality.

SD-square epoch job27912 passed in7m36s. Training takes48.438,97.005,
193.810 and388.078seconds for1/2/4/8 epochs. All four use512 distinct
records. The one-epoch parameter fingerprint exactly reproduces the
selected earlier fit. Final-epoch mean training KL is0.39231,0.27866,
0.20265 and0.12539 across the four separately scheduled fits. These
are training losses, not held-out or decoding improvements. The audit
rebuilds the selection and raw fitting gate. Curated sd-square-epochs.json
contains source-bound checkpoints and per-epoch losses.

Immediate next work: compare all four epoch endpoints under the corrected
SD-square inference precision on the same sixteen development prompts,
using two disjoint eight-request jobs to stay inside pilot limits. Reuse
benchmark_sd_square.py and check_sd_square_campaign.py with an epoch
campaign builder. Include runtime-local AR and the independent drafter.
Then freeze the epoch endpoint before the predeclared longer-output
pilots. PARD's next required path is eight2048-token requests and matched
AR with per-request verification replicas, before the full quality batch.
The raw/curated public-runtime results are now in the29-page paper,
whose main text remains nine pages. Every technical audit passes. Only
the new/adjacent pages26--27 were visually inspected here, so full
color/grayscale review remains pending and paper completion is not claimed.

## Epoch comparison and capped quality continuation

Jobs27913/27914 passed the common epoch decoding comparison in6m14s and
5m58s. All96 rows (six methods, sixteen disjoint requests) reproduce.
At1/2/4/8 epochs the selected KL steering reaches18.561/18.452/18.390/
18.258tokens/s versus18.750 for the independent drafter. Ratios are
0.990/0.984/0.981/0.974; the last interval is[0.949,0.9997]. More fitting
lowers training loss without establishing a decoding improvement.
Freeze the one-epoch checkpoint before longer quality evaluation.

PARD quality pilot27915 passed in1m40s, both methods8/8 correct. Full
quality27916 then passed in22m53s: PARD105/128 versus matched eager AR
103/128,105.626 versus31.668tokens/s, ratio3.335[3.222,3.451]. PARD has
8 cap hits versus AR11. Its conservative paired accuracy difference
interval is[-0.06342,0.09301], so the one-point margin is not established.
All256 raw rows, separate observed replicas, pinned scorer and source
checks reproduce. This is development data, not untouched confirmation.

SD-square selected-epoch quality pilots27917/27918 passed in3m09s/2m33s.
Both methods score8/8, with20.934 versus12.559tokens/s. Raw gates,
training/inference identities and pinned scoring reproduce. Proceed
with the predeclared128-request quality batch at the same2048-token cap,
using two sequential64-request shards on four GPUs. Each proper run
has a45-minute Slurm ceiling and2640-second process ceiling. This extends
evaluation only; no new fitting or dataset expansion occurs. The full
builder requires the audited matching eight-request pilot and freezes
its checkpoint and runtime identities. Pilot config regeneration remains
byte-identical after this extension. Expected total evaluation duration
is roughly40--60minutes based on the pilot, with output lengths variable.

Before any final confirmation generation, the quality inference protocol
was amended to use a conservative paired exact interval; see
2026-09-06-paired-quality-inference.md. The1pp margin,256 primary requests,
256 reserve and no optional reserve extension remain unchanged. The
previous protocol is archived and the local exposure audit found no
held-out/reserve ID matches. Sparse concordant pilot results must not
be treated as proof of noninferiority through a degenerate bootstrap.

Large-data scaling remains paused. All current jobs are quality evaluation
of frozen models; future fitting stays within512 or2048 distinct examples.

Full SD-square initial deployment27919 failed before inference in7seconds
because the vendor archive was extracted at the project root. Its
27920 dependent job was cancelled without allocation. Preserve the
failed raw packaging artifacts and ledger. A clean replacement directory
extracts vendor files under vendor/sd-square and verifies every pinned
Python source hash before submission. Replacement jobs27921/27922 use
the same immutable source26f10fb and unchanged experiment configs.
Collector478038 covers the replacement jobs. No numerical retry or
protocol change was involved.

The30-page manuscript now contains raw-audited SD-square epoch costs
and common decoding, plus PARD full capped quality. Main text remains
nine pages and42 citation keys resolve. All technical audits pass.
Page27 was visually inspected in color after rebuilding: tables, text,
line numbers and boundaries are legible with no clipping. Full color/
grayscale review of all30 pages remains pending, so overall QA staysFAIL.

SD-square replacement full job27921 failed after9m46s in an unscored
16-token warmup on global request26, math500:test/prealgebra/1044.json.
Preserve39 partial rows and the traceback without a full-quality claim.
Dependent27922 was cancelled. Public source has PAD_FACTOR4 and stops
when the next nine-slot speculative cycle would exhaust its physical
buffer, so cap16 permits only seven cycles. Low acceptance can therefore
stop warmup before16 tokens without EOS. This is a hypothesis until the
bounded GPU reproduction27925 verifies it with the frozen checkpoint.
Diagnostic caps16/64 use immediate and deferred observers on requests
24--27. Actual verifier equality remains strict, and normal scored
requests retain their original EOS/cap completion requirement. Unit tests
cover the physical guard, ordinary cap/EOS and unexplained early stops.

Diagnosis27925 passed in1m01s and its gate reproduces from all four raw
worker traces. Failed request26 yields15 tokens in seven cycles atcap16,
exactly the public physical-slot guard. Atcap64 it reaches64 counted
(71 raw) tokens in20cycles. Every immediate/deferred trace is identical,
with actual verifier agreement and unchanged inherited weights.

Implement a prospective warmup-only policy: retain the public16-token
warmup request and record cap, EOS or its explained physical-slot guard.
Unexpected stops still fail. Measured2048-token generation keeps the
strict EOS/cap check and every warmup trace is recorded and reverified.
This changes the harness's treatment of an unscored public return, not
public generation statements or the checkpoint. New eight-request pilot
uses fixed development indices24--31, including the failing request.
The original eight-request pilot/config and result reproduce unchanged.
A guarded full batch requires both the audited diagnosis and the matching
new pilot; it cannot be promoted solely from the earlier pilot.

Guard-policy pilot jobs27928/27929 use source759b0a3 and four GPUs each,
with ten-minute ceilings and afterok sequencing. They cover development
indices24--31 at2048 scored tokens with runtime-local AR. Collector487770
is active. Do not use39 partial rows from27921 in the final128 comparison.
