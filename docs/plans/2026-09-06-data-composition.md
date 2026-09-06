# Matched-count data composition

Keep2048 distinct fitting examples per arm. Compare the existing2048
Numina math records with an alternating mix of their first1024 records
and1024 general-instruction records. Do not describe the latter as
strictly non-math: the general source can contain mathematical questions.

Use databricks/databricks-dolly-15k at revision
bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a, with pinned JSONL SHA
2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec.
The publisher's dataset card supplies category labels and CC-BY-SA3.0
licensing: https://huggingface.co/datasets/databricks/databricks-dolly-15k.
Downloaded source provenance is composition-data/source.json. Preserve
attribution and license in any redistributed derived data. Do not claim
benchmark independence beyond the actual exact/lexical filter.

Select by deterministic source-stratified hash order, before fitting or
decoding. Exclude normalized exact and5-shingle Jaccard>=0.6 matches to
all current evaluation prompts, the complete reference MATH test set,
historical fitting records, current math train/validation records, and
confirmation/reserve prompts. Only the deterministic overlap filter may
read confirmation text; do not inspect outcomes or generate candidates.
Reserve1024 general records for fitting validation before taking1024
for the mixed training arm, filtering training against that validation.
Common validation for both arms consists of existing1024 math-validation
records plus those1024 general records. Report loss separately by domain.

Use the same192-token cap, target/proposer interface, seed1729,8192 updates,
batch4 and objective for dense, linear512 and MLP512 at DFlash8B. This is
six primary fits. Two additional dense seed1730 controls quantify reference
variability. Refit math-only arms under the same common validation and
verify the original final mapper weights where a matching seed/config is
available. Validation must not select a different training horizon. Report
actual non-padding training tokens because equal records and updates do
not ensure equal token computation. Retain per-domain length/category
counts, training/validation errors, and inherited extraction cost.

Run a new under-ten-minute cache/fit/decode pilot before each proper
new path. Compare all endpoints together on exposed math, code and
conversation development inputs with runtime-local AR/source controls.
Scored quality and final confirmation remain separate. This is not a
large-data scaling resumption or an autoresearch proposal.

Built the manifests at data/composition/math-dolly-v1. All three files
contain exactly2048 distinct records. Common validation has1024 math
and1024 general records, disjoint from both training pools. The filter
removed11 duplicate instructions and14 near matches before completing
the selected general pools. Raw data stays outside Git; the manifest gate,
source download hashes, deterministic builder and license attribution
are retained. No composition fitting or decoding has occurred yet.
Implementation of data-root/manifest selection in cache extraction and
per-domain validation reporting remains required before its GPU pilot.

A second independent local build reproduces every selected JSON manifest
and its audit gate byte-for-byte. This verifies deterministic selection
and overlap filtering for the pinned inputs. It does not replace the
required cache/gradient/export/decoding pilot or per-domain fitting metrics.

Before any composition fitting, version 2 changes only common-validation order
to alternate math and general-instruction records. Version 1 placed all math
records first, so the 16-record compatibility pilot would miss general-domain
validation. Training files are byte-identical to version 1; validation membership
and counts are unchanged. Preserve manifest-gate-v1.json as the earlier prototype.
The active data directory is data/composition/math-dolly-v2; its validation SHA
is 4581ca42c4c4faa3eda1b2d4802178dafc29fee9c1083b2fba4c336e578d7037.
The version 2 builder also writes attribution with the derived data.

Cache extraction now accepts an explicit data root and training filename.
The pilot records both manifest hashes, the manifest-gate hash, available counts,
and domain policy. Full extraction must match these exact inputs; an old Numina
pilot cannot authorize a custom composition cache. Domain diagnostics reuse the
same per-record forward passes and preserve aggregate record weighting. Labels
are cache metadata and do not alter tensor payloads or the training objective.
Tests cover input mismatches, domain coverage, count limits, and unchanged metrics.

The eight proper fits and exact-cache resource pilots are declared in
configs/submission/scaling/composition-small-v1. Each arm uses four independent
fits: dense, factorized 512, MLP 512, and a second dense seed. Save the four-pass
and fixed-exposure checkpoints (2048 and 8192 updates); per-update training loss
still records the complete learning trajectory. Full validation runs initially
and at those checkpoints. Compatibility and exact-cache pilots must pass before
proper fitting; no proper composition fitting has been launched yet.

The task-difficulty analysis is separately integrated into the31-page
paper on page24. All technical audits pass and main text remains9pages,
with42 resolved citation keys. The new page was visually inspected in
color without clipping or overlap. Full31-page color/grayscale review
remains pending, and the paper is not declared submission-ready.

Compatibility pilots 27944 (math) and 27945 (mixed) are queued from source
ce00aabae4bf5c936042b0fe5bdbedafeb62c2f7 at
/home/aryama.murthy/relayspec-composition-ce00aab. Job 27944 follows completion
of the existing 14B fit chain (afterany:27943); 27945 requires 27944 success.
Both held submissions were checked for exactly four requested GPUs before
release. Each uses 64 training and 16 validation examples for infrastructure
checks only, with a 540-second process timeout. Jobs and raw collection are
tracked in composition-pilots/jobs.json. Proper extraction/fitting is not yet
submitted. Eight focused local tests, Ruff, shell syntax, and the version 2
byte-for-byte rebuild checks pass; see composition-data/implementation-checks.json.

The shared cached-fit evidence auditor now checks per-domain diagnostics when
requested: every declared validation domain must be present, domain record/token
counts must sum to aggregate coverage and remain unchanged across checkpoints,
and domain metrics must reconstruct the aggregate with record weighting. Missing
labels, nonfinite values, coverage drift and incorrect averaging fail explicitly.
Eighteen focused evidence/domain/campaign tests pass. The unchanged legacy branch
also re-audits the eight collected 14B fits successfully. No GPU rerun is needed
for this local evidence check, and no composition result is yet claimed.
