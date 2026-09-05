# Data, capacity, and fitting research plan

User-requested extension, 5 September 2026. This supplements the evidence execution
plan; unfinished correctness, baseline, breadth, stability, and artifact tasks
remain in scope. TriSpec is excluded from the requested comparison study.

## Execution order and evidence

1. Finish and collect the live continuous job 27583, zero-training pilot 27673,
   and nested data jobs 27675–27677. Preserve their immutable snapshots. Combine
   64/128/256 with completed 512/1024/2048 and continuous-step-1024 (4096).
   Fix updates at 1024 and presentations at 4096. Report the smallest tested
   set within 5% of the best throughput, with paired uncertainty and scores;
   this development analysis cannot establish a universal minimum. If 64
   qualifies, extend to 16/32; if the boundary is between points, test the
   intervening geometric midpoint. Confirm the selected boundary and full-data
   reference with seeds 1730/1731 and separate evaluation requests.
2. Capacity: dense linear reference and factorized linear ranks
   64/128/256/512/1024/2048/4096; GELU MLP widths at exactly the same values.
   Both factorizations have h*(input+output) parameters, allowing matched
   capacity comparisons. Rank above output width increases parameterization,
   not linear function-class rank. Report dense parameter count, all fitted
   and deployed parameter counts, bytes, mapper latency, total decoding cost,
   acceptance, quality, fitting time, and peak memory. Hold taps, normalization,
   objective, records, updates, and seeds fixed within each comparison. Execute
   the crossed matrix below; endpoint screening does not replace the required
   data-by-capacity interaction study.
3. Regularization: distinguish explicit L2, lambda/2 * sum(weight**2), from
   AdamW decoupled weight decay. Explicit L2 lambda grid 0/1e-7/1e-6/1e-5/1e-4;
   AdamW decay grid 0/1e-4/1e-3/1e-2, separate arms with no simultaneous penalty.
   Screen dense, matched rank-512 linear and MLP-512, then confirm useful cells.
   Log data loss and penalty separately; retain failures and nonfinite runs.
4. Larger data and overfitting: build nested 8192/16384/32768 distinct training
   manifests from the same eligible source population, with unique IDs and
   content hashes, and a disjoint validation manifest. Audit exact/near overlap
   with evaluation, including solutions. Repetition never increases distinct
   count. Compare dense and selected small/large MLP at fixed updates and at
   fixed passes as separate panels. Record train and validation interface loss
   through continuous trajectories and evaluate selected checkpoints in actual
   decoding. Training loss alone cannot establish overfitting. Measure token
   counts and source/task composition, not just record counts. Current 4096
   manifest cannot supply these larger-data experiments.
5. Model/task complexity: repeat the selected data/capacity boundary for DFlash
   14B and EAGLE-3 8B/14B, keeping family-specific interfaces. Report task,
   length, target scale, and source-target relationship separately. Finish
   breadth scoring, seed controls, direct-feature controls, cheap adaptation
   baseline, and output-divergence diagnostics from the evidence plan. Preserve
   all cells where a relay slows down or loses accuracy.
6. After fixed baselines, run adaptive research rounds. Each idea has a written
   hypothesis, parent run, single intervention, expected outcome and falsifier.
   Candidate order: residual correction, relative/cosine/scale-aware losses,
   then verification-aware losses if interface error fails to predict acceptance.
   Freeze each round before observing its results. Keep a ledger of every
   attempt, wall time, source/config hashes, outcome, and promotion decision.
7. Promote confirmed results into one hashed registry, regenerate manuscript
   tables/figures and claim map, run software/evidence/paper checks, compile and
   inspect the PDF. Publish negative and inconclusive results with scope. No
   claim of an optimum or completed paper until the relevant evidence exists.

## Required crossed study: data × mapper complexity

The central question is how the data needed for useful decoding changes with
mapper capacity and hypothesis class. Separate one-dimensional sweeps cannot
answer it. Distinguish three axes: mapper complexity, target/drafter identity,
and training-data composition. Target parameter count alone is not a controlled
measure of task difficulty or transfer difficulty.

### A. Predeclared core matrix

Start with DFlash 4B → Qwen3-8B. Every cell uses the same frozen models, tap
locations, normalization, data order, evaluation requests and tokenization.

| Axis | Required values | Purpose |
| --- | --- | --- |
| Distinct fitting records N | 128, 512, 2048, 8192, 32768 | Five approximately logarithmic data levels, including genuinely larger data |
| Mapper family | Factorized linear, two-layer GELU MLP | Compare function class at equal parameter count |
| Factorization rank / MLP width h | 128, 512, 2048 | Small, intermediate and large capacity crossed with every N |
| Dense reference | Original single linear map at every N | Establish whether factorization/nonlinearity helps the deployed baseline |
| Extended capacity sweep | h = 64, 256, 1024, 4096 at N = 512 and 32768, both families | Locate saturation and test larger/smaller parameterizations |

The core has 35 cells (5 data levels × 7 architectures); the extended sweep adds
16, for 51 distinct settings per optimization-budget regime and seed. Existing
runs count only if data IDs, optimization budget, implementation and evaluation
match. The earlier 64–4096 dense sweep remains a separate, finer minimum-data
study. Failed and timed-out cells stay visible in the matrix.

With input width d and output width k, dense has d*k parameters; each factored
linear/MLP pair has h*(d+k). Report these actual counts, effective linear rank
bound min(d,k,h), FLOPs and bytes. Increasing h beyond k does not expand the
linear function class; any gain there concerns optimization/parameterization.
A factored linear map can be multiplied into a dense matrix for inference.
Benchmark factored and collapsed deployment separately where useful, charge
collapse time, and validate numerical/output differences before comparing speed.

### B. Separate data benefit from optimization benefit

Use two explicitly different panels, never pooled into one scaling curve:

- **Fixed exposure budget:** 8192 optimizer updates × global batch 4 = 32768
  record presentations for every core cell. Every N is actually visited;
  smaller sets repeat. Save at 128/512/2048/8192 updates to expose convergence.
  This holds updates/presentations fixed, not GPU seconds: mapper FLOPs differ.
- **Fixed passes:** four passes over each nested set, batch 4, hence N updates.
  Data and computation both grow here. Plot performance against both N and
  measured training time. An optional equal-wall-time panel uses its own label.

One 1024-update run with batch 4 can see at most 4096 distinct records; it cannot
support an 8192–32768 distinct-data claim. Record available records, records seen,
record presentations, nonpadding tokens, optimizer updates and elapsed time
separately at every checkpoint. Larger-data pilot/feature-cache measurements
must establish the actual time budget before submitting the full matrix. Stage
jobs in batches; the matrix is required work, not a blind all-at-once launch.

### C. Validation and overfitting protocol

Create a fixed 1024-record fitting-validation split and a fixed 256-record
training diagnostic subset; neither changes across architecture cells. Preserve
source/task/length proportions across nested training sets. Where the source
cannot supply 32768 eligible distinct records plus validation, record the gap
and obtain a documented eligible extension before launching that level.

Keep fitting validation, decoding development and confirmatory evaluation roles
separate, with IDs/hashes and overlap audit before selection. At each saved
checkpoint, evaluate the same unregularized interface objective on train and
validation subsets under identical masks and normalization, plus relative MSE,
cosine error and feature-norm error. Report the gap and both losses, not a gap
alone. Measure development proposal acceptance and paired end-to-end decoding at
predeclared checkpoints; fit loss is not a speed or quality surrogate by default.

Overfitting evidence requires improving train loss accompanied by worsening
validation loss or held-out decoding behavior as fitting continues. Underfitting
requires persistently poor train and validation behavior after checking optimizer
convergence. If larger MLPs fail, distinguish inadequate optimization from lack
of data by learning-rate checks and longer continuous trajectories, using the
same search budget for matched linear controls. Do not call an undertrained MLP
intrinsically worse. Add L2/weight-decay comparisons at low and high N for dense,
linear-512 and MLP-512; analyze whether regularization shifts the data boundary.

### D. Confirmation across target models and drafter families

Use DFlash-8B development results to freeze three capacities (small,
intermediate, large), not only the winner. Repeat both mapper families and the
dense reference at N = 512, 8192, 32768 for DFlash-14B, EAGLE-3-8B and EAGLE-3-14B.
This is 21 cells per target/drafter setting and budget regime. Retain each
family's correct interface and matched data. A family-specific budget adjustment
must be declared and reported rather than hidden in a pooled curve.

Run the primary core grid first with seed 1729. Confirm the three data anchors
128/2048/32768 across the seven core architectures with seeds 1730/1731; confirm
inferred minimum-data boundaries and selected deployment settings in the other
model/family panels with the same seeds. If rankings reverse, expand neighboring
cells before claiming an optimum. Show seed variation separately from paired
request uncertainty. Validate final choices on requests never used for tuning.

Evaluate all selected settings across math, code and conversation workloads.
Within workloads stratify by input length, output length and an independently
specified difficulty label where available; do not define difficulty using the
candidate's own success. Add a matched-count/math-only versus mixed-domain
training comparison at N = 2048 and 8192 for dense and matched linear/MLP-512.
This tests data diversity separately from data quantity. Report exact composition,
length distributions and training tokens; a longer example is not equal compute.

### E. Questions, plots, and completion evidence

| Question | Required analysis/artifact |
| --- | --- |
| How much data does each capacity need? | Learning curves against distinct N, separate panels by architecture/budget; smallest tested N reaching 95% of that architecture's best development throughput, then independent confirmation |
| Which mapper is best at each data budget? | N × parameter-count heatmaps for validation error, acceptance, AR-relative throughput and task score; preserve missing/failure cells |
| Does more data rescue large MLPs? | Matched linear/MLP contrasts across N, train/validation trajectories, seed variation and interaction contrasts |
| Is apparent saturation statistically resolved? | Paired request intervals for speed/quality differences plus seed results; report unresolved plateaus, not a unique optimum |
| What is the practical deployment choice? | Pareto plots of throughput/quality versus fit GPU-seconds, deployed bytes and mapper latency, with AR/native/source controls measured in the same evaluations |
| Does the conclusion transfer? | Faceted curves for each target and drafter family, and task/length/composition breakdowns including regressions |

The 95% threshold is an operational development criterion, not universal
optimality. Report two thresholds: relative to each architecture's best and
relative to the best tested architecture overall. Require quality noninferiority
with a margin frozen before confirmatory scoring, and report confidence intervals;
a point estimate inside 5% alone is insufficient confirmation. If confidence
intervals are too wide, the threshold remains uncertain. Treat power-law fits as
exploratory only if enough non-saturated points support them; do not extrapolate
an unmeasured optimal data size.

Completion requires hashed manifests/configs, completed or explicitly diagnosed
cell outcomes, training/validation trajectories, saved checkpoints, paired raw
decoding results, scorer provenance, uncertainty analyses, and regenerated paper
figures/tables. Pilot success and a filled scheduling ledger are not completion.

## Four-GPU scheduling and trial gate

Use Turing's existing account/partition and at most four GPUs simultaneously.
Current jobs use all four as DDP training workers and independent paired-eval
workers. Do not overlap a second allocation with them. Use length-aware batching
only after verifying masked losses and equal effective data/gradient accounting;
benchmark throughput before changing an established comparison protocol.

Every new architecture/loss path starts with a Slurm 10-minute limit including
loading and a 540-second process timeout. Fit 16 updates, save/reload, verify
finite gradients/weights, and evaluate eight paired requests. Passing this is an
infrastructure gate, not scientific evidence. A directional trial within the same
10-minute ceiling uses a predeclared development subset and budget; promotion
requires a measurable validation/acceptance improvement or useful memory/cost
tradeoff, without a quality red flag. Timed-out trials are recorded, not silently
extended. Proper runs follow only successful gates and use full declared data,
matched controls, 2048-token evaluation cap where applicable, and seeds.

Autoresearch must not select on the confirmatory evaluation set. Stop a branch
after three failed directional trials or two successive rounds without a useful
Pareto improvement; report the boundary found. This is a stopping rule for each
branch, not permission to omit the fixed requested experiments. Reuse feature
caches only with model/tap/tokenizer/data hashes and measured storage budgets.

## Current completion status

Historical pending jobs, seven fixed-work data sizes, continuous checkpoints,
zero-training controls and AR-paired code scoring have completed and been
collected. Factorized/MLP fitting, explicit L2, corrected normalization in
validation, expanded disjoint data, hashed frozen-feature caches, and four
independent cached GPU workers are implemented. The full32,768/1,024 feature
cache passed its gate. The width64/4096 pilot passed in2m17s.

The matrix-v2 declaration preserves all required cells and both budget panels,
adds explicit epoch checkpoints, and saves final optimizer/RNG state for longer
training. Full MLP512 and matching linear512 trajectories are starting as
jobs27726/27727. The remaining matrix, regularization, convergence/learning-rate
checks, seeds, family/task/data-composition replication, cheap-adaptation
baseline, untouched evaluation and bounded autoresearch remain required.

Completed historical data and code-quality evidence is now in the manuscript
appendix with regenerated, source-validated tables/figures. The draft compiles
with nine main pages. Full scientific completion and the final PDF review are
still pending. This status does not establish an optimal mapper or dataset size.

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


The executable post-pause declaration is now
`configs/submission/scaling/matrix-focused-v1/matrix.json`: N512/2048,
30 primary capacity cells,42 nonzero regularization cells and four dense seed
confirmations, with four prior matching fits reused. See
`reports/mapper-scaling-20260905/focused-results.json` for explicit completed
and missing cells. More-epoch continuations use the same smaller fitting sets
and are separately declared in `continued-small-data.json`. All new jobs
remain bounded by successful pilots; these declarations do not claim completion.
