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
