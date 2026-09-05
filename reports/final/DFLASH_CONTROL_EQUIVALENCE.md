# DFlash native-control reuse audit

The full target-autoregressive and released target-specific DFlash-8B controls
are reused from the immutable Part-1 run rather than spending roughly another
native-AR GPU-hour on outputs unaffected by the selected relay checkpoint.
They are contextual deployment-frontier controls, not the denominator of the
causal claim.

The control and final configs have identical:

- Qwen3-8B/14B target IDs and immutable revisions;
- Qwen3-4B DFlash proposer ID, immutable revision, and source-code commit;
- non-thinking chat formatting, greedy temperature zero, top-p one, top-k
  disabled, 2,048-token cap, BF16/SDPA path, seed, block size 16, and four-GPU
  hardware;
- MATH-500 prompt and answer population.

The older manifest names problems by original MATH file path and the newer
manifest names them by deterministic index, so their raw manifest hashes
differ. After whitespace normalization and order-independent sorting, all
500 prompt hashes intersect and all 500 answers agree. The canonical sorted
`(prompt, answer)` population SHA-256 is
`ee01ac953478587a5e49f1f565056a3fd8e44e6f99e2e7ab0c5392af1f8ab062`
for both manifests.

The harness source hash changed because the final run added the optimized
34-layer provider, richer instrumentation, gates, and second-family support.
Therefore native-control timing is presented as same-hardware context only;
all headline source/relay speedups, confidence intervals, accuracy deltas, and
component fractions come from paired rows within the new run. This avoids
claiming cross-run timing as the causal estimate.

