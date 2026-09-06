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
