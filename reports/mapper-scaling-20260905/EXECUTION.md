# Mapper scaling execution, 5 September 2026

Status: active research; the paper and full requested study are not complete.
Code lives in branch `research/scaling-autoresearch-20260905` at
`/home/aryamavmurthy/work/RelaySpec-scaling`. Historical result collection lives
in `/home/aryamavmurthy/work/RelaySpec/reports/`.

## Latest state, 23:50 IST

- Full frozen-feature cache27723 completed in12m22s:32,768 train records,
  1,024 validation records and292,810,789,056bytes. Cache SHA-256 is
  a2fd012df1da5554f8e211cf0b60aec098d3754d866bc50e9623f3117f43f8e5.
- Epoch/continuation capacity pilot27725 passed in2m17s. It exercised widths
  64 and4096 for both linear and MLP models, streaming the full cache, fixed
  1024-record validation, BF16 equivalence and saved-optimizer round trips.
  Pilot27724 was cancelled while still pending and consumed no GPU time.
- Full MLP512 data trajectories are running as27726 (primary-04), one per GPU
  at N128/2048/512/8192. Matching linear512 trajectories27727 follow. Both
  use immutable source13ab419 and matrix-v2, preserving explicit epoch
  checkpoints and both budget panels. Job ledger:matrix-wave1/jobs.json.
- Cached/full/scoring collectors run detached on the workstation. Revalidate
  their PIDs and Slurm before relying on these observations.

## Verified work

- Commits `86b1557`, `194ba65`, `308d32d`: factorized linear maps, matched-width
  MLP loading in both benchmark families, explicit L2 distinct from AdamW decay,
  training/validation diagnostics, bounded pilots/directional trials, larger
  nested data builder, and exact-task-subset EvalPlus wrapper.
- 228 CPU tests passed (3 manuscript tests excluded), with lint/format checks.
  This establishes software checks, not paper readiness.
- Continuous run 27583 and unfitted pilot 27673 recovered and scored.
  Continuous 128/256/512/1024/2048 updates: 3.65/4.34/4.67/4.90/5.03x AR;
  each scores 105/128 on development-exposed MATH requests.
- Data64 job 27675 completed and scored: 3.55x AR, ~72.8% of observed-best
  mapper throughput, 105/128 correct. It fails the descriptive 95% threshold.
  The fixed-compute 512/1024/2048/4096 cells remain within 5% of observed best.
- Data128 27676 completed and was collected/scored: 4.10x AR, ~83.7% of
  observed-best throughput, 105/128 correct. This also fails the 95% threshold.
- Original pilots 27685–27689 completed in roughly one minute each. Their
  training/checkpoint/decoding outputs are valid infrastructure evidence, but
  fitting-validation metrics are INVALID because they omitted DFlash's final
  normalization. Do not interpret them as overfitting evidence.
- Correction `194ba65` applies the exact released `draft.hidden_norm` in
  diagnostics. Pilots 27694–27698 use a separate immutable v2 snapshot.
  All five corrected pilots passed checkpoint/completeness/diagnostic checks in
  62–69 seconds. Directional jobs 27711–27715 are queued behind data256; each
  uses 1024 updates and the fixed <=10-minute trial budget.
- Data256 27677 follows the corrected pilot chain. No active benchmark was
  cancelled. Every GPU job uses all four physical L40S GPUs on node07.

## Time and GPU policy

Historical 128-request evaluations generate four arms up to 2048 tokens.
The 512-example fit took 98 seconds in a 22m34s allocation; AR generation alone
used 11m26s on the busiest worker. Search pilots use 8 requests / 128 tokens;
directional trials use 16 / 256 and 1024 fitting updates. Slurm limit is 10 min;
process timeout 540 seconds. Capped outputs are not final task-quality evidence.

Next efficiency work: reuse frozen feature caches after an equivalence pilot,
fit independent candidates in parallel within one four-GPU allocation, and
benchmark multiple candidate maps in one paired campaign with a shared,
rotated AR/native/source reference. Avoid repeating identical AR generations
for every candidate. Do not splice historical timings into new paired claims.

## Expanded data

`data/scaling/manifests-pinned/` contains nested 128/512/2048/8192/16384/32768
training sets, fixed 1024 validation and 256 training-diagnostic records.
Source is immutable NuminaMath-CoT revision
`9d8d210c9f6a36c8f3cd84045668c9b7800ef517`, train shard 0/5, restricted to
`cn_k12`, `synthetic_math`, `orca_math`. This is a separate data-source panel,
not a continuation of the historical MATH-only curve. The pinned first shard
contains 119604 records in these three strata before filtering.

The builder checks immutable source-file hashes, excludes all original MATH
test problems/solutions, historical fitting problems/solutions and current
benchmark prompts, then excludes validation from training. It uses normalized
exact matches and 5-shingle Jaccard >=0.6, and removes duplicate problems.
For the full set it excluded 17 exact matches, 527 near matches and 252 duplicate
problems. Lexical checks do not establish semantic/template independence.
Source mixture, counts, file hashes and code hashes are in `data/manifest-gate.json`.
The 32-record preparation pilot took 4.3 seconds; full preparation took 50 seconds.
Raw parquet files and 90 MB of generated manifests stay outside Git.
Remote data destination:
`/scratch/aryama.murthy/factorspec-runtime-20260826/relayspec/scaling-data/numina-v1`
(compute-node view; login uses `/scratch/node07/aryama.murthy/...`).
Verify rsync completion and hashes before feature extraction.

## Code-quality scoring

CPU-only pilot sequence on node01:
- 27692 failed: original scorer environment lacked `datasets`.
- 27704 failed: dependency transfer had not completed; `libarrow_substrait`
  was unavailable. Transfer is now complete. Do not use this run.
- 27705 failed: official EvalPlus requires all tasks; a two-task pilot is a
  subset. No quality result is claimed from this attempt.
- 27707 passed the corrected subset wrapper in 2m14s under a 10-minute limit, using
  two HumanEval and two MBPP tasks for all three saved EAGLE-3 methods.

The wrapper preserves every official base/plus test for selected IDs, rejects
unknown or duplicate IDs, uses a separate subset ground-truth cache key and
records scorer/data/sample hashes. It adds no missing-task placeholders.
Pinned upstream EvalPlus commit: `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`.
CPU source snapshot: `/home/aryama.murthy/relayspec-scoring-20260905`.
Dependency overlay: `/home/aryama.murthy/relayspec-eval-data-deps-20260905`;
package versions are in `scoring-data-dependencies.txt`.
Full CPU scoring jobs 27708 (8B) and 27709 (14B) are running with eight CPUs
each on node01. No GPU generation rerun is used. Conversation quality remains
a separate unresolved task.

## Next actions

1. Revalidate live Slurm jobs and durable collector PIDs (do not trust stale
   status files). Collect 27677 when finished and update the minimum-data report.
2. Collect and interpret directional trials 27711–27715 after data256 finishes.
   Corrected pilot gates already passed; directional outcomes remain pending.
3. Collect full code-scoring jobs 27708/27709; preserve failures and provenance.
4. Implement and pilot cached feature fitting with record-weighted masked
   batching equivalent to global batch four, then shared-reference multi-map
   decoding. Cache extraction cost must be reported separately from fitting.
5. Execute the crossed scientific matrix, regularization, larger-data/longer
   training trajectories, seeds, family/task replication, cheap-adaptation
   baseline, and bounded autoresearch from the detailed plan.
6. Promote only audited evidence, regenerate figures/tables and manuscript,
   compile/inspect the PDF and perform the complete submission audit.

No universal minimum/optimum, completed autoresearch campaign, or paper-ready
status has been established.

## Verified update, 23:24 IST

- Commit b7efe32 implements hashed frozen-feature extraction, independent cached
  fitting on four GPUs, and shared-reference multi-mapper decoding. 239 CPU tests
  passed (3 manuscript tests excluded). End-to-end pilot 27722 passed in 1m51s:
  64 train / 16 validation records, four simultaneous 16-update fits, cache
  integrity and BF16 gradient-equivalence gates, and exact duplicate-mapper
  decoding/acceptance isolation. It is plumbing evidence, not a capacity result.
- Full data256 job 27677 completed in 22m41s; fitting took 97.93s. It achieved
  4.542x AR for decode-only timing (4.512x for end-to-end request timing). The complete single-seed historical curve's smallest tested set
  within 5% of the best measured throughput remains 512. Independent confirmation
  is outstanding. All seven data sizes are collected and scored.
- Dense directional 27711 completed in 3m18s (97.04s fitting); factor512 27712
  completed, with MLP and regularized directional trials progressing.
- CPU code quality jobs 27708/27709 completed and were collected. CODE_QUALITY.md
  and code-quality-ar.json contain official EvalPlus base/plus AR-relative paired
  outcomes and uncertainty. 8B MBPP plus loses four passes; 14B gains two.
- Full cache extraction is now a separate gated stage. Pilot estimates about
  300 GB for 33,792 records; scratch has about 14 TB free. Extraction cost and
  model-loading wall time are logged separately and charged to the study.

## Epoch-aware fitting and paper update

All five 1024-update directional jobs27711-27715 completed in2m58s-3m18s
and were collected. Their source-hashed aggregate is directional-summary.json.
Dense/factor512/MLP512 end-to-end speedups are4.610/2.867/2.526x AR on16
development requests capped at256tokens. L2=1e-6 did not improve the fixed
validation objective. These are early-budget single-seed outcomes, not a
verdict on MLP capacity or generalization.

User steering added explicit epoch1/2/3/4 and later power-of-two checkpoints
within the two fixed budget regimes (matrix-v2). Optimizer and RNG state
are preserved at each trajectory endpoint for further fitting when justified.
Convergence/learning-rate checks remain required before interpreting poor MLP
results. Source13ab419 includes these changes.

The manuscript appendix now includes the completed seven-point historical data
curve, five-point continuous curve, and8 AR-paired code-quality comparisons.
New tables/figure are generated with raw-source hash checks. The draft compiled
to20pages with main text ending onpage9 and42 resolved citations. Changed
pages16-17 were rendered and inspected. The full color/grayscale visual-review
record is stale and remains pending the final paper build. New Numina capacity
results, seeds, family/task transfer and autoresearch are still outstanding.

## Large-data batch queued, 23:56 IST

Job27728 follows27727 and fits dense, MLP2048, factorized4096 and MLP4096
on32,768 distinct records, each through four epochs (32,768updates).
The same trajectory includes the8192-update fixed-exposure checkpoint.
Its ledger and independent collector are matrix-large-data/jobs.json.
The full-cache path, source13ab419 and successful capacity pilot27725
remain fixed. The current MLP512 job27726 has reached7424/8192updates
on its last, streaming-data worker; the three smaller-data fits are ahead.
Do not infer all matrix cells are complete from these initial batches.

Measured execution suggests grouping remaining batches by cache I/O as well
as parameter count: small preloaded-data workers finish before the larger
streaming worker. Preserve scientific cell definitions while improving
scheduling. A GPU-resident small-data cache or work queue needs its own
short equivalence/resource pilot before changing the fitting path.

## First full MLP data curve completed, 23:58 IST

Job27726 passed in8m08s. Each MLP512 fit used8192updates/32768presentations.
At N128/512/2048/8192, fixed1024-record validation objective was respectively
0.37380/0.30818/0.28776/0.28184. The corresponding epochs were256/64/16/4.
Training diagnostics were0.24254/0.26714/0.28123/0.28295, using min(N,256)
training-pool records. These single-seed fitting results support further
large-data study, but do not establish full-answer speed or quality gains.
Source-gated checkpoints/trajectories are in initial-mlp-data-curve.json.

The smaller fits' loops took123-147seconds; the N8192 streaming loop took
440seconds, including375seconds recorded in input I/O. This measured cost
should guide the next scheduling/cache optimization pilot. Matching linear
job27727 is now running and large-data27728 remains queued behind it.

## User scope update, 6 September 2026

Further large-data scaling is PAUSED at the user's request. Do not restart
the32,768-example job27728 or submit new large-data/four-epoch replicas.
Preserve the completed128--8192 fitting curves and the verified full cache.
The partially run27728 is a stopped attempt, not a completed scientific cell.

Current priorities are capacity, matched linear/MLP comparisons, explicit L2
and AdamW decay, convergence/learning-rate checks, seeds, model/family
replication, actual decoding/quality and the other pending paper baselines.
Use512 and2048 distinct fitting records for the focused capacity and
regularization study. Existing8192 outcomes can be analyzed, but further
large-data expansion is deferred. This user change supersedes the earlier
requirement to execute all32768 cells now.

Cache-access pilot27730 is an implementation speed/equivalence check using
4096 records for mapped I/O and512 for GPU residency, not a new32k data
scaling experiment. Keep its10minute ceiling; its results may accelerate
the remaining smaller-data fits.


## Focused execution after the large-data pause

The exact smaller-data declaration is `configs/submission/scaling/matrix-focused-v1/matrix.json`.
It contains30 capacity settings at N512/2048 (dense and linear/MLP widths
64/128/256/512/1024/2048/4096), reuses four completed width512 fits from27726/27727,
and declares42 separate nonzero L2/AdamW arms plus four dense seed confirmations.
Thus72 new fits remain in18 four-GPU batches. Submit in measured stages, not a
blind queue. No new8192/32768 data cells are in this declaration.

Cache-access pilot27730 completed in2m05s and passed bit-identical weights,
optimizer states and all logged losses. Mapped streaming improved loop time
106.73s to68.87s (1.55x); GPU residency improved19.73s to13.41s (1.47x).
New optimized fits require the same-cache exact-equivalence gate and capacity
gate. New batch scripts default to540seconds/10minutes, including fitting,
validation and checkpoint verification. Full raw pilot outputs are collected.

The next decoding check uses eight completed factorized/MLP512 checkpoints,
N128/512/2048/8192, with shared AR/native/source controls on16 MATH requests
capped at256tokens. This analyzes existing data and does not resume large-data
fitting. It remains a development throughput/acceptance diagnostic, not a
full-answer quality result. CPU validation:16 targeted tests and lint passed.


## Focused first wave and continuation implementation

Job27736 completed eight-map decoding in2m20s. The N8192/N2048 paired throughput
ratios were1.0165 (95% CI0.9859--1.0495) for factor512 and1.0342 (1.0123--1.0615)
for MLP512, on16 requests capped256tokens. Keep the limited protocol explicit;
large-data fitting remains paused. All raw outputs and scoring are collected.

Job27737 completed four dense fits in2m59s; fitting-loop times106.8--112.6s,
input preparation1.8--1.9s. N512 validation objectives across seeds1729/1730/1731
were0.21008/0.21018/0.20985; N2048 seed1729 was0.18055. These are fitting results,
not yet dense decoding comparisons. Width64/128 batches27739/27740 follow.

The collector now supports cache-access and continuation pilot gates. Exact
continuation restores optimizer, mapper, RNG, token count and cyclic sample
position, rejecting changes to data, seed, loss, learning rate or architecture.
The bounded pilot compares1024 uninterrupted updates with512+512 for dense
and MLP512 with explicit L2, checking every loss and final state bit for bit.
Promotion requires this exact-cache pilot gate. CPU validation:17 targeted tests
and lint passed. The GPU continuation pilot has not yet run.


## Continuation verified and smaller-data work resumed

Continuation pilot27741 passed in1m11s: dense and MLP uninterrupted1024-update
fits were bit-identical to512+512 resumed fits for weights, Adam state, RNG,
token accounting, every training loss and common validation checkpoints.
The verified gate is continuation-pilot/verified-gate.json; raw artifacts are
collected. The focused registry now contains16 completed cells out of76,
including four reused fits and three N512 dense seeds. The study is incomplete.

Job27742 continues the four original width512 factorized/MLP fits at N512/2048,
from8192 to32768 total optimizer updates. This is MORE EPOCHS ON THE SAME SMALL
DATASETS, not32768 distinct examples. The explicit config is
configs/submission/scaling/continued-small-data.json; saved optimizer and exact
parent hashes are required. It has the10minute job/540second process ceiling.
Widths256 and1024 follow as27743/27744. All use source d34d85f and four GPUs
sequentially; collector346496 handles this immutable wave. Widths2048/4096,
regularization, remaining confirmation and full decoding are still pending.
No new large-data scaling jobs have been submitted or resumed.


## Longer-training results and second-family preparation

Job27742 completed in4m24s. The N2048 MLP validation objective improved
0.28776 ->0.28261 ->0.27867 at8192/16384/32768 updates; matching linear512
improved0.26853 ->0.26577 ->0.26407. At N512 the MLP improved0.30818 ->0.30668
->0.30595, while linear512 validation worsened0.28617 ->0.28623 ->0.28733
as training loss fell0.24431 ->0.23943 ->0.23500. This is limited single-seed
fitting evidence consistent with mild overfitting in the smaller linear arm,
not a quality/speed claim. Parent endpoints match resumed diagnostics exactly.
Source-hashed trajectories are in continued-small-data-results.json.

Widths256/1024/2048/4096 are jobs27743/27744/27745/27746. Larger-data expansion
is still paused. Shared-reference EAGLE-3 mapper campaigns and configurable
family cache pilots are implemented for the required family checks. EAGLE-3
candidates use fresh per-request context providers and expose checkpoint hashes;
its duplicate gate checks outputs, accepted progress and call counts. New
feature extraction defaults to2048 examples and10minutes and requires the
corresponding configuration's pilot. 34 targeted CPU tests and lint pass;
EAGLE-3 GPU pilot remains required before full replication.


## Primary grid complete, AdamW gate and EAGLE launcher repair

All30 primary capacity cells are completed and collected (plus two additional
N512 dense seeds,32/76 focused-matrix cells total). Width2048 batch27745 took
2m51s; width4096 batch27746 took4m18s. Dense has lower validation error than
all alternatives at8192updates. At N512, increasing MLP width2048->4096 reduces
training error0.15857->0.12779 but worsens validation0.24384->0.25007; at N2048
both improve, including validation0.19819->0.19362. These single-seed validation
results motivate regularization and optimization checks, not a universal claim.

AdamW resource/decoding pilot27747 passed in1m39s. Gate and raw artifacts are
collected. EAGLE cache pilot27748 failed in16s before training: its launcher
set DEEPSPEC_PYTHON_OVERLAY but omitted that directory from PYTHONPATH, causing
an unrelated Gemma4 import from DeepSpec to fail in the default Transformers
installation. Existing working EAGLE jobs include the pinned overlay. The
three new family launchers now select their import paths from the actual YAML
proposer family and reject a missing overlay. No dependency/version downgrade
or silent model change was made. Failed logs/config are retained in the original
repo under eagle3-cache-pilot/failed-27748. The bounded EAGLE pilot must rerun.

Unstarted DFlash decoding27756 was cancelled by its failed EAGLE dependency.
Its independent replacement27760 uses the identical config and already-gated
DFlash source, followed by regularization batches27761/27762. The33-method,
16-request256-token capacity campaign loads all30 fitted maps and shares
AR/native/source references. Its gated builder refuses incomplete primary
checkpoints; no main-quality conclusion is implied by this development test.

New source-checked capacity and epoch figures/table are integrated into the
appendix. The21-page draft compiles without overfull boxes or unresolved
references; pages17--19 were rendered and inspected. Full PDF color/grayscale
review is still pending the final manuscript. The asset auditor now checks
both historical and new capacity builders against their recorded raw evidence.


## All-primary decoding completed

Job27760 passed in6m40s with528 rows (30 candidates plus3 controls on16 requests).
Native target reached189.08tok/s, dense N512/2048 reached184.10/185.62tok/s,
factor1024 N512/2048 reached180.15/182.95tok/s, factor4096 N2048 reached186.86tok/s,
and MLP1024/4096 N2048 reached177.61/180.18tok/s. These are256-token diagnostics.
All capped scores were2/16, so full-answer evaluation remains necessary.
The factor1024 N512 point reaches96.4% of the highest mapper point with23.59M
parameters; this is a descriptive development boundary, not a confirmed optimum.
Paired candidate intervals and raw hashes are in capacity-decoding-results.json.
Regularization27761 completed in2m28s and27762 is running normally.


## Second-family pilot passed and current continuation

EAGLE-3 pilot27777 passed in2m07s after the explicit dependency-path repair.
It verified64 training/16 validation cache records, four16-update fits, BF16
batch gradients and eight-request duplicate-map EAGLE decoding (identical
outputs, progress, call counts and checkpoint hashes). Raw outputs and scoring
are collected. The successful gate is eagle3-cache-pilot-v2/verified-gate.json.
Job27783 will extract only2048 training/1024 validation records under a10minute
ceiling, using source36104a0 and the exact pilot training-config hash. It follows
regularization27778/27779/27780, which continue the already-validated DFlash path.
The independent branch waits for an allocation to end without requiring a
scientifically unrelated family pilot to succeed.

Eight regularization cells have completed (40/76 focused cells total). For
factor512 at N2048, L2 coefficients1e-7/1e-6/1e-5/1e-4 give validation
0.26970/0.27392/0.28619/0.31197 versus0.26853 with no penalty. AdamW values
1e-4/1e-3/1e-2 give0.268525/0.268539/0.268701, essentially unchanged at this
seed and budget. Do not generalize these initial linear results to MLPs.
The remaining regularization batches05--10 are not submitted yet.

Paper capacity assets now read the frozen primary-capacity-results.json so
live regularization updates to focused-results.json do not invalidate a figure
whose primary data are unchanged. The paper audit passes every technical/source
check and remains incomplete only for the final full color/grayscale PDF review.

Independent evaluation work: all existing GSM8K evaluation manifests contain
128 already-evaluated questions. A scan of collected benchmark-rank files
found128 unique GSM8K IDs. The complete pinned test Arrow file is already
cached at /home/aryamavmurthy/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow.
The next confirmation manifest should use fresh questions from that source,
exclude recorded evaluation prompts plus fitting/validation exact and near
matches, and freeze the selection before observing new model outputs. This
inventory alone is not an untouched-evaluation or semantic-independence gate.
Do not reuse the old MATH confirmatory filename as evidence of current non-exposure.

Next work: collect27783, generalize run_cached_batch's capacity pilot campaign
config (currently DFlash-only) and its Slurm family import paths, then pilot
selected EAGLE capacity extremes on the new small cache before full replication.
Continue the regularization matrix, selected learning-rate/seed checks, full
answer-quality evaluations, other pending paper baselines, and bounded
hypothesis-driven research after fixed baselines. Large-data expansion remains
paused. Neither the focused matrix nor the overall paper is complete.


## Small-data continuation and second-family capacity declaration

Large-data expansion remains paused. EAGLE cache27783 passed in1m35s,
containing2048 training and1024 validation records (26.67GB, index hash
9c18d186420ff569186fc1232b411cbb984ae05bbe09564a36f6179ceaccad19).
DFlash regularization27785/27786 completed in2m22s/2m16s. Batches07--10
are jobs27787--27790, chained through the validated DFlash path, using
at most four GPUs. Sixty of76 focused cells are currently collected and verified.

The new matrix-eagle3-small-v1 declaration contains14 primary fits:
N512/2048 crossed with dense and factorized/MLP widths128/512/2048, plus
two additional dense N2048 seeds. Each uses8192 updates and the EAGLE
scale-preserving input convention. A separate four-candidate16-update
pilot checks the width extremes, BF16 batch gradients and duplicate-map
decoding before full runs. Host-buffer caching avoids promoting an optimization
whose exact-cache equivalence gate exists only for DFlash. The batch driver
now checks campaign target/proposer identity against cache metadata and
selects family-specific controls and dependency paths. No new EAGLE fits
have run at this declaration point.


## Longer-answer development decoding declaration

The EAGLE capacity pilot is job27793, source5aad553, queued after the
regularization allocation releases. No full EAGLE capacity fits are promoted
until that exact-cache pilot passes.

The new build_small_data_quality_campaign.py verifies nine completed
small-data checkpoints and their fit/cache gates before emitting a four-request
pilot and128-request development configuration, both capped at2048 output
tokens. They compare dense N512/2048, factor1024 N512, factor4096 N2048,
MLP4096 N2048, and factor512/MLP512 N2048 at8192 and32768 updates, plus
AR/native/source references. The latter continuation uses the same2048
distinct examples. This is decoding only. It tests whether longer training
changes deployed speed or answer quality. These exposed MATH prompts are
development data, not independent confirmation. Cap hits must be reported.
The full128-request campaign remains gated on the sub-ten-minute pilot
and its measured runtime. Generated configurations rebuild byte-identically.

Longer-answer pilot27795 is queued after EAGLE pilot27793, using immutable
sourceeafdbd6 and a540-second process limit/ten-minute allocation. Its collector
is running independently. The128-request full evaluation has NOT been submitted.
Regularization27788 completed in3m13s,27789 is running and27790 is queued.
The latest verified registry contains64/76 cells; collection may lag Slurm.
All14 MLP512 penalty cells are already collected. At seed1729 and8192
updates, no tested L2/AdamW penalty improves either N512 or N2048 endpoint
over its zero-penalty baseline. These are fitting-validation results only.
Remaining immediate work is to collect the final focused matrix and the two
new pilots, review their resource/correctness gates, then promote only the
matching successful small-data paths. Large-data fits remain paused.


## Confirmation data and learning-rate declaration

A new GSM8K confirmation manifest freezes256 requests plus a separate256
reserve from the pinned1319-question test Arrow cache. The reverse lexical
index was checked against individual existing overlap audits, including
multiple matches and short duplicates (three tests passed). The scan covers
381 collected rank files, all existing evaluation manifests,32768 cached
Numina fitting problems/solutions,1024 validation records and4096 historical
fitting records. It excludes the128 previously evaluated GSM8K questions;
1191 candidates survive the declared exact/near rule and within-set filtering.
This is an audit of existing text, not more large-data fitting. The gate
records all input/output hashes and does not claim semantic independence or
global non-exposure. The data builder uses the existing .venv-data interpreter,
which supplies pyarrow; the ML test interpreter does not contain it.

The confirmation analysis freezes a3-percentage-point quality margin versus
matched AR and95% throughput retention versus a frozen best development
mapper, with paired95% intervals and10000 request bootstrap samples. The
selected checkpoint/config/scorer record is still required before generation.
Neither confirmation nor reserve outputs have been generated.

The matrix-small-lr-v1 declaration adds eight bounded learning-rate checks:
width4096 factorized/MLP, N512/2048, rates2e-4 and1.8e-3 versus the existing
6e-4 trajectories, fixed8192 updates and validation1024. These test whether
the wide MLP gap reflects optimizer settings or convergence. Every batch
retains the ten-minute ceiling and existing exact-cache/width correctness
gates. These fits have not started at declaration. Large-data expansion is
still paused.


## Focused matrix complete, pilot promotions and paper update

All76 focused fitting cells are now complete and source-verified, including
42 nonzero penalty cells and the four additional dense seed fits. The largest
AdamW endpoint improvement is0.165% relative validation error for linear512
N512. No tested L2 improves a matched zero-penalty endpoint, and no tested
penalty improves either MLP512 endpoint at seed1729. The new source-checked
regularization figure/table are in the appendix. The22-page PDF compiled,
and new pages19/20 were rendered and visually inspected. Every technical
manuscript audit passes, including generated-asset regeneration and the
nine-page main-text boundary. Full22-page color/grayscale review remains
pending and is explicitly not marked passed.

EAGLE capacity pilot27793 passed in2m39s, including four width-extreme fits
and eight-request identical-checkpoint decoding equivalence. It promoted
the declared16 small-data family fits as27799/27800/27801/27802, using
source5aad553, host-buffer caching and at most four GPUs. These follow
learning-rate trials27796/27797 (source0c63a4c).

Longer-answer pilot27795 passed in1m58s with48 paired rows, nine selected
small-data maps, AR/native/source controls, and zero2048-token cap hits.
Every candidate checkpoint hash matches the frozen declaration. The128-request
development evaluation is now job27806, sourceeafdbd6, queued after27802.
The pilot projects28.77 minutes of generation. Its process cap is60minutes
and Slurm ceiling65minutes to accommodate longer requests. This time is
existing-checkpoint decoding, not mapper fitting. Raw/scored outputs are
collected by independent durable watchers. This development evaluation does
not consume the new GSM8K confirmation/reserve requests.

Remaining work includes collecting these studies, selected target/family
replication, stronger seed/deployment-boundary checks, composition/workload
analysis, the cheap-drafter-adaptation control, then bounded hypothesis-driven
research after the fixed baselines. Freeze the final model selection before
confirmation outputs. The paper and overall research goal remain incomplete.
Large-data expansion remains paused throughout.


## Drafter-adaptation implementation and optimizer results

The eight wide-mapper learning-rate cells are complete. At N2048, reducing
the learning rate from6e-4 to2e-4 lowers width4096 MLP validation error
from0.193615 to0.191160 (1.27%) and factored linear from0.184454 to0.182866
(0.86%). Raising it to1.8e-3 worsens the MLP endpoint. N512 wide MLPs show
earlier validation minima, so the final-update ranking alone is insufficient
for architecture claims. These fitting results still need decoding checks.

The new bounded DFlash drafter-adaptation pilot is implemented, not yet run
at this declaration. It reuses64 cached training records and a verified
dense N512 mapper checkpoint at128 initial updates. Four GPU workers compare
connector-only target-greedy CE, frozen-connector drafter LoRA ranks8/32,
and a duplicate rank32 seed. It uses16 additional updates, batch4, lr2e-4,
q_proj/v_proj adaptation and alpha2r. Target-greedy labels are teacher-forced
on the final15 positions of a16-token block. No source transformer forward
is needed during fitting after frozen inputs are prepared.

The pilot checks zero-LoRA forward identity, finite nonzero gradients,
inherited drafter weights unchanged, independent merged-weight export/reload,
duplicate-seed fitted weights, zero-update decoding identity and duplicate
adapted-drafter decoding. The benchmark driver isolates each adapted drafter
from inherited controls and records both mapper and drafter hashes. Nineteen
CPU tests pass, including FP32/BF16 export/reload cases. The pilot ceiling
is ten minutes including fitting and decoding. Full matched-compute/data
budgets remain unexecuted and must charge connector initialization, label
preparation and fitting separately. SD2/PARD comparisons and second-family
adaptation remain separate pending baseline requirements.

Drafter-adaptation pilot27810 is queued using source45504e8. It follows
EAGLE fit27802 and precedes the longer development evaluation27806. The
latter was held only while its allocation-order dependency was changed,
then released, and depends on afterany27810 so a failed independent pilot
cannot invalidate its already-passed DFlash decoding gate. No running job
was interrupted. A durable collector handles the new adaptation_pilot kind.

The capacity-campaign builder now accepts an explicit family template and
chooses matching AR/native/source references. It checks the template hash
where the fit gate records one and the family's declared input normalization.
The original DFlash30-map configuration and provenance reproduce byte for
byte. EAGLE capacity decoding remains gated on all14 declared primary fits.


## Confirmation quality-margin correction before generation

The initial new confirmation protocol used a3-point quality margin, which
was looser than the existing evidence plan's1-point tolerance. Corrected
it to0.01 absolute accuracy and retained the original protocol hash in a
pre-generation amendment. This supersedes the earlier3-point entry above.
No selected-model record or confirmation/reserve generation exists yet.
The fixed256-request test may be inconclusive, which must be reported.
Do not loosen the criterion or add reserve requests after seeing results.
The data manifests and their hashes are unchanged.

## Small-data continuation and failed adaptation diagnosis

Reconfirmed the user's pause on further large-data fitting. No32,768-example
fit is running or queued. Preserve existing scaling results and feature caches.
Continue at512/2048 distinct fitting examples and assess actual decoding gains.

All16 EAGLE small-capacity fitting cells are now collected. The complete
14-primary-map decoding configuration was generated only after verifying
every fit gate, declared trial, cache hash and8192-update checkpoint.
It uses16 development requests,256 output tokens and three family controls.
The EAGLE path already passed its short correctness pilot27793.

Adaptation pilot27810 failed after45seconds at the outer duplicate-seed
merged-weight comparison. Each of its four individual fitting gates passed.
The two rank32 workers have identical first-step loss4.539029240608215,
then differ from step2 onward. This does not yet distinguish initialization,
prepared inputs or backward numerical nondeterminism. The failed logs,
fit gates and loss traces are preserved in the original report tree.
No adaptation result was promoted or decoded.

A diagnostic-only variant records exact prepared-input, teacher-label,
initial-trainable, initial-logit and first-step gradient hashes. It repeats
the first backward pass with identical weights and restored RNG, records
within-worker gradient/logit agreement, then restores the original gradients
and RNG before the original optimizer update. The equality gate remains
unchanged. This gathers boundary evidence before choosing a fix, and its
instrumented timing is not a scientific training-cost result. Nineteen
existing adaptation, LoRA and mapper-campaign tests and targeted lint pass.

Queued diagnostic27811 afterany27806 and EAGLE decoding27812 afterany27811
from immutable source5d44306. Each requests four L40S GPUs with a540second
process timeout and ten-minute Slurm ceiling. The dependency chain keeps
total allocation at four GPUs and preserves the independent quality run.
The EAGLE correctness pilot27793 already passed. A durable collector watches
the new wave. Diagnostic failure must retain its boundary evidence for root
cause analysis and cannot promote adaptation. No large-data run was queued.

## Convergence decoding and selected seed replication declaration

Declared20 mapper checkpoints for paired16-request256-token development
decoding: all12 width4096 factor/MLP trials atN512/2048 and rates2e-4,
6e-4 and1.8e-3, all six distinct validation-selected early checkpoints,
plus denseN2048 and factor1024N512 controls. Three inherited references
share the same requests. This tests whether improved feature loss or early
stopping improves decoding. Neither this cap nor this exposed development
set supports a full-answer confirmation claim.

The builder verifies raw fit/validation/gate hashes, the exact declared
trials, cache identity and every selected checkpoint hash. It recomputes
the validation minimum from raw trajectories before selecting checkpoints.
The complete focused registry is used because the historically frozen
primary-only registry intentionally predates completed penalty cells.
Repeated builds yield identical20-checkpoint declarations and a deliberately
changed validation hash is rejected. Targeted lint passes.

Declared eight further fits, seeds1730/1731, at four previously tested
points: factor1024/N512/rate6e-4, factor4096/N2048/rate2e-4,
MLP4096/N2048/rate2e-4 and MLP4096/N512/rate2e-4. Reference seed1729
already exists. Report fixed8192-update endpoints and each seed's best
saved validation checkpoint separately. The N512 MLP arm checks whether
the early validation minimum repeats across seeds. All fitting remains
at512/2048 distinct examples on the same cached features. These are
replications of pilot-verified fitting paths, not new-data experiments.

Convergence decoding is queued as27817, followed by selected seed fits27818
and27819, immutable sourceeb650a3. Each has a540second process cap and
ten-minute Slurm ceiling. Slurm returned nonconsecutive job IDs. The seed
dependencies were corrected while both jobs were still pending, and every
final dependency was verified through scontrol before recording the ledger.
No extra GPU allocation occurred. A durable collector watches all three.

## Public PARD baseline pilot implementation

Pinned AMD-AGI/PARD code6f279bf3f1680e0b5d71c562ca5b91bdeef4c038 and
released amd/PARD-Qwen3-0.6B revisionf9f650fbab180c26498817718f0db5cae8f25136.
The public1.50GB weight blob is checked against its LFS SHA256 before use.
The baseline performs zero new-target fitting. Its inherited training cost
is separate from checkpoint preparation and inference.

The adapter verifies the exact upstream source hash and inserts request
timing and raw-token capture into the public generation loop. An AST test
proves every original statement remains in its original order. The original
reported TPS is not used: its timer is reset after the first verification
block, while its numerator includes that block's tokens. Our measured
request interval includes cache reset and both prefills, ending before
detokenization. Raw cap/EOS overshoot remains recorded, while output counts
include only the requested prefix without subtracting computation time.

This first setting uses the public uncompiled eager-attention/static-cache
path and a matched eager AR control on identical pretokenized user prompts.
It does not claim a cross-engine RelaySpec speed ranking. Four independent
GPU workers each evaluate one128-token request, with matched16-token warmups
excluded. PARD must match its duplicate and matched AR outputs, and its
duplicate proposal trajectory must match. Raw rows are written before a
failed equality gate. A CPU-only Slurm stage prepares an isolated pinned
Transformers4.51.3 inference overlay and public checkpoint on scratch.
Both setup and GPU pilot have ten-minute ceilings. No full PARD run is
promoted before the four-worker gate passes. Sixteen targeted tests pass,
including three PARD instrumentation/stop-boundary tests, plus targeted lint
and shell syntax checks. The GPU integration is not yet verified here.
