# Small-data target replication

The user's large-data pause supersedes the older8192/32768-example
replication grid. Keep512/2048 distinct fitting examples. This experiment
changes target size and mapper capacity, not the training-set scale ceiling.
The declared protocol is configs/submission/scaling/target14b-small-v1/protocol.json.

Reuse run_cached_fit_pilot.py and cached_fit_pilot.sbatch for each14B
family. Its64-example,16-update cache/gradient/export/duplicate-decoding
check is a compatibility pilot, not a new scientific data-scaling cell.
Both pilots retain the ten-minute ceiling and run sequentially after
SD-square quality so the total allocation never exceeds four GPUs.

Only after each exact training-config pilot passes, extract2048 fitting
and1024 validation records with extract_scaling_cache.py. Reuse the fixed
Numina manifest prefix, max192tokens and target/family-specific interface.
Then fit dense plus factorized and MLP widths512/1024/4096 at512/2048
examples,8192 batch-four updates, seed1729. The anchors are fixed from
the8B capacity evidence before14B fitting. Match wide-model rate checks
and selected seed replication after observing the initial decode grid,
retain every tested setting, and avoid calling an underoptimized fit a
function-class limitation. Do not infer task quality from feature loss.

DFlash14B uses correct AR/source controls and a matched dense map because
the pinned local inventory has no native14B DFlash checkpoint. Do not
substitute the8B native proposer under a14B label. EAGLE14B includes its
existing pinned native proposer. Preserve DFlash input normalization
and EAGLE scale preservation. Report each family's input dimensions,
training positions, fitting cost and matched throughput separately.

This track does not replace remaining EAGLE trainable baselines,
composition/complexity analysis, bounded autoresearch, untouched
confirmation, final manuscript review or anonymous reproduction.

Queued DFlash14B pilot27923 and EAGLE14B pilot27924 from immutable
sourcee1df8bad6c5a3687776f783c1264f52b720351b2. Each requests four GPUs
and ten minutes. DFlash starts after SD-square full-quality shard27922
terminates, and EAGLE follows DFlash. The two pilots are independent,
so afterany sequences resources without treating one family's failure
as evidence about the other. Full cache extraction still requires its
own matching pilot to pass. Existing mapper/caching tests:14 passed.
Pinned14B target and native EAGLE snapshot directories exist on node07,
with ample scratch capacity. Collector481273 records both pilot outcomes.

Both compatibility pilots passed: DFlash27923 in2m31s and EAGLE27924
in2m37s. DFlash dense map has65,536,000 parameters. Both gates bind
the correct14B target/source proposer,64 training and16 validation
records, four finite16-update fits, and eight exactly matching duplicate
map requests. Native14B EAGLE remains separate from source reuse.

Prepared32 proper fitting cells across both families:14 primary cells
per family plus dense512/2048 seed1730 reference-variability controls.
These extra seeds are scientific controls, not duplicated GPU padding.
The original seed1729 grid and all anchors remain unchanged. Matrices
use the existing trial validator and four-GPU batching. Host-buffer
cache access stays enabled until any optimized access has an exact pilot
on the new cache. Capacity pilots at widths512/4096 precede proper fits.

Queued bounded extraction27926 (DFlash) and27927 (EAGLE), using the passed
matching pilots. Each extracts2048 training and1024 validation records,
with at most ten minutes and four GPUs. They follow termination
diagnosis27925 sequentially. Collector486037 handles both cache outputs.

Extraction27926/27927 both passed in1m59s, producing32,589,626,496bytes
each and exactly2048 train/1024 validation entries. DFlash index SHA is
6db523b44d7cac89e9b59cb250dea29a6c06729d2381a9d486d9d23b8da91c07.
EAGLE index SHA is
c35ea12d2164a5a3576667f05824ecf91d8b5829dd84ae80d764d375127ffdb2.
Source-bound pilot and extraction gate links match. Actual shards and
index are revalidated by fitting before use. Capacity pilots27931/27932
use source759b0a3, widths512/4096 and only16 updates. They run sequentially
after the guarded SD-square pilots27928/27929; collector488196 covers
both. No proper fit is launched before its exact cache capacity pilot.

Capacity pilots27931/27932 passed in3m23s/3m31s, with matching cache-index
and trial hashes. DFlash width4096 fits use about3.29GB peak GPU memory.
Its16-update optimizer time is0.61--0.63seconds, separate from validation,
cache loading, checkpoint export and decoding. These costs support the
bounded proper8192-update batches, whose completion remains unproven.

Queued DFlash fitting27936--27939 and EAGLE fitting27940--27943 from
source759b0a3. Each batch has four unique cells, four GPUs and a ten-minute
allocation/540-second process limit. The first DFlash batch follows
SD-square full quality27935, then each family uses afterok sequencing.
EAGLE begins after the DFlash chain terminates, independently of its
success. All runs recheck the exact-cache capacity pilot and source-bound
trials before fitting. Collectors494794/495278 cover DFlash/EAGLE.
The32 cells include28 primary fits and four dense seed1730 controls.
Do not report a fitting or replication result before collecting its gate.

## Endpoint decoding preparation

The completed-cache and capacity-pilot gates do not establish that the 32 proper
fits have completed. Audit each family with scripts/audit_target14_fits.py after
all four corresponding jobs finish. It requires all 14 primary cells and both
dense seed controls, checks immutable job source/trial/campaign hashes, binds
the exact-cache pilot, reconstructs every update and checkpoint-validation
record, and derives parameters from the actual interface dimensions. Missing
or mismatched evidence must prevent a completed registry.

Both 14B interfaces have input width 25,600 and output width 2,560. The eight
completed resource-pilot trajectories pass the new raw-log audit: width 512
has 14,417,920 parameters and width 4096 has 115,343,360 for both factored linear
and MLP. Those 16-update pilots are infrastructure evidence only; do not call
them proper fitting results or extrapolate validation performance from them.

Use the existing build_capacity_campaign.py with --include-seed-controls and
--fit-registry to declare each family's complete 16-endpoint comparison. The
registry must match the matrix and every underlying source hash before the
builder can write a configuration. The campaign remains 16 common exposed
MATH requests at a 256-token cap with runtime-local AR/source controls, plus
pinned native EAGLE where available. Fixed 8192-update endpoints are selected
regardless of feature-validation minima. The additional dense seeds are not
excluded for unfavorable fitting or decoding results.

The 14B cache/gradient/export/decoding path has already passed its bounded
compatibility and exact-cache capacity pilots. Run the complete endpoint
comparison with a ten-minute allocation/540-second process limit first. Do not
extend a failed observation timeout without checking the actual job and error.
Full-answer quality must still have its separate short 2048-token pilot before
the larger quality evaluation. Code/conversation breadth, wide-model rate
checks, selected third seeds, EAGLE trainable baselines and untouched final
confirmation remain required where specified by the overall plan.

Once the fits finish, the exact local command sequence for each FAMILY is:

```
PYTHONPATH=src ../RelaySpec/.venv/bin/python scripts/audit_target14_fits.py --family FAMILY --raw-root ../RelaySpec --output reports/mapper-scaling-20260905/target14b-FAMILY-fit-results.json
PYTHONPATH=src ../RelaySpec/.venv/bin/python scripts/build_capacity_campaign.py --raw-root ../RelaySpec --matrix configs/submission/scaling/target14b-small-v1/matrix-FAMILY/matrix.json --template configs/submission/scaling/target14b-small-v1/campaign-FAMILY-pilot.yaml --fit-registry reports/mapper-scaling-20260905/target14b-FAMILY-fit-results.json --include-seed-controls --output configs/submission/scaling/target14b-small-v1/campaign-FAMILY-capacity.yaml
```

Replace FAMILY with dflash or eagle3; these commands do not submit or restart
jobs. Deploy the generated, committed configuration with its provenance only
after audit success. Record actual Slurm IDs and keep the total at four GPUs.
The registry/campaign integrity and existing mapper tests passed (22 tests),
including rejection of omitted seed controls and subsequently changed raw
fitting evidence. The full proper-fit audit is still pending actual job output.

Prepared scripts/audit_target14_capacity.py for the forthcoming endpoint runs.
It binds all 16 declared endpoints, the complete fit registry and raw input
hashes, campaign source/config, exact common 16-request membership, correct
answer references and mapper hashes. Saved raw output is independently rescored
with the pinned scorer. It computes paired comparisons against AR, source reuse,
available native EAGLE and the seed-1729 dense2048 reference. The 256-token scores
remain short-output diagnostics, not full-answer quality or noninferiority.
The script passes formatting, lint and CLI import checks; validation against
actual 14B capacity output remains pending. No new decoding result is claimed.

DFlash14B jobs 27936--27939 all completed in 4m59s, 5m09s, 6m50s and
8m26s. The full source/cache-bound audit passes all 16 trajectories, including
all 8192 updates, validation checkpoints, actual dimensions and token exposure.
The curated result is target14b-dflash-fit-results.json. Both dense seeds agree
closely: validation objective about 0.244 at N512 and 0.204 at N2048.
These are fitting results only, not a decoding ranking or final-quality claim.

Generated campaign-dflash-capacity.yaml and its provenance after the completed
registry. It contains all 16 endpoints and two controls (AR and source reuse),
16 common development requests and a 256-token cap. Every endpoint remains fixed
at 8192 updates. Schedule its bounded ten-minute comparison after the existing
four-GPU chain, then run audit_target14_capacity.py before using any result.

Queued DFlash capacity decoding 27954 from source e744b57, at
/home/aryama.murthy/relayspec-dflash14-capacity-e744b57. It follows 27952
with afterany resource sequencing, has a ten-minute allocation and 540-second
process limit, and was released only after verifying its held four-GPU request
and deployed config/source hashes. Collection/scoring is tracked by
reports/mapper-scaling-20260905/target14b-capacity-dflash/jobs.json.

The DFlash14 fitting report and validation-trajectory figure now regenerate from
the audited registry with scripts/summarize_target14_fitting.py. Two complete
builds reproduce the Markdown, PNG and PDF hashes; the plotted page was inspected
in color and grayscale. The 512-example dense and wide maps show small late
validation increases, while every 2048-example fit has its best saved validation
at the fixed endpoint. Wide MLP512-data loss rises from 0.27389 at4096 updates to
0.27839 at8192; the 2048-data endpoint reaches0.21467. This does not select a
decoding checkpoint or establish a unique optimum. Preserve full trajectories,
including step0, and separate training-seed from request uncertainty.
