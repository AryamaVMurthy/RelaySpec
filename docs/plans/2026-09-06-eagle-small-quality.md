# EAGLE small-data answer quality

Large-data scaling stays paused. This evaluation reuses completed EAGLE-3 8B
mapper fits from the fixed 512/2048-example grid; no new training is required.
The source-verified fitting registry and endpoint campaign must reproduce before
building the quality configuration.

The selected eight endpoints are dense, factorized width2048, and MLP width2048
at both512 and2048 examples, plus factorized width512 and MLP width512 at2048.
All use their fixed8192-update endpoint. Selection follows exposed256-token
capacity decoding and precedes this2048-token quality evaluation. It is not
untouched candidate confirmation. Width128 remains in the complete capacity
study, with its measured weaker throughput; it is not a quality finalist.

Compare all eight with runtime-local AR, the pinned native8B EAGLE drafter and
source reuse. First run8 requests at2048-token caps under the540-second process
and10-minute Slurm limits. Inspect completion, raw scorer replay and resource
cost before launching proper evaluation. The proper comparison has128 exposed
MATH requests, split into two disjoint64-request shards. Request membership is
identical to the DFlash small-data quality comparison. Inverse-permuted manifest
records preserve the original shuffled request order within each shard under
the unmodified runtime selector. Method rotation restarts per shard, declared
in advance. A deterministic selection test and actual-config verification pass.

The declaration is configs/submission/scaling/eagle3-small-quality-v1/protocol.json.
The builder reuses summarize_focused_scaling.collect and existing endpoint
provenance; selected checkpoint hashes must match GPU fit gates. Each full shard
uses exactly the same methods, checkpoints, target/proposer, generation settings
and scorer. Do not pool cross-runtime TPS as an algorithm-only comparison.

scripts/audit_eagle_quality.py checks exact request/method coverage, source and
config hashes, answer references, mapper identities and output caps. It re-runs
the original pinned scoring script on temporary copies of raw outputs and
requires identical scores and analysis. Preserve original raw files. Report
paired throughput and conservative exact discordance intervals against AR and
dense2048, with the existing0.01 quality margin. These are individual development
comparisons, not simultaneous guarantees after selection. Inconclusive remains
inconclusive. Cap-length outputs may include EOS at the cap and must not all be
labeled truncations. Full output requires the matching audited pilot result.

Neither generated configs nor local checks establish a successful quality run.
Only the bounded pilot is eligible for initial submission. EAGLE trainable
baselines, target14B replication, composition, final confirmation, manuscript
integration and full visual review remain in the overall goal.
