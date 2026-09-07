# transfer.zip reproduction

Archive SHA256: de405a43604c6e939f2bfd345be7e85acc16b70241ff8504ad9432075b3956e8.

Unmodified archive extracted to external/transfer-reproduction/transfer. MANIFEST.sha256 passes; all10 bundled CPU tests pass. No trained model/checkpoint is included. Reproduction has not yet produced a mapper or new benchmark result.

## Recipe corrections relative to prior loss ablation

| Setting | Prior ablation | Archive recipe |
|---|---|---|
| Distinct training examples |512|16,384|
| Sequences |192-token cached supplied-solution prefixes|8B-generated continuations up to4,096 output tokens|
| Token positions |32 fixed positions per record|25% segment-stratified sampling across prompt and response|
| Initialization |PyTorch Linear default|Xavier uniform|
| Optimizer schedule |lr0.0006 constant|lr0.001,5% warmup,cosine decay|
| Fitting budget |128/1,024 updates|3 fixed epochs, historically7,935 updates|
| Seed |1729|42|
| Inference |RelaySpec Transformers runtime|vLLM0.28.0+cu129, batch-invariant evaluation|

The prior negative result is not a reproduction of this recipe. Both results must retain their own scope.

## Execution

1. Job28613 installs exact pinned dependencies into an isolated scratch environment and runs pip/environment/tests checks.
2. Job28618 completes the exact pinned installation using persistent download caches if the initial30minute allocation is insufficient; it requests2CPUs and no GPUs. Job28614 depends on successful environment verification, reuses only exact pinned base/drafter HF snapshots, downloads the pinned dataset shard, and recreates/verifies split and evaluation prompt hashes.
3. Job28615 depends on successful preparation. Four independent <=540second operational probes: eight training continuations, AR short evaluation, native short evaluation, and synthetic GPU mapper math/folding verification.
4. Inspect actual generation samples and gates before the full two-worker generation. Preserve original queue windows and128 active sequences. No historical outputs used as generated training data.
5. Compare rollout hashes. Any mismatch must be inspected and reported before deciding whether only a new-data realization can be reproduced.
6. Run sampled/dense capture checks, inspect and approve, then capture both models on identical complete generated sequences.
7. Train exactly3epochs, compare mapper tensors to historical hashes, export and verify folding.
8. Run exact evaluation pilot and then the128-prompt protocol with the original worker partition. Report cross-pipeline and historical token equality separately from fresh timing.

Original recorded generation summed across workers: 11643.1 seconds, approximately97.0 minutes wall time with two balanced workers. Original fit887.5seconds. Feature payload336GiB. These are historical measurements, not completion promises. Scratch preflight found13TiB free.

GPU budget: maximum4 across active GPU stages. Scheduler auto-adds1 billed GPU to the4CPU setup/preparation requests; those stages do not perform GPU computation. Every operational probe has540second cap. Full generation must retain two long-lived workers to preserve the archive scheduling contract.
