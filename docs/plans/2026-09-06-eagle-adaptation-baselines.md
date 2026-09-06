# EAGLE trainable baseline replication

Large-data scaling stays paused. Extend the existing DFlash connector-CE versus
frozen-connector drafter-LoRA pipeline to EAGLE-3 8B. Reuse the pinned source
proposer, cached target features, trainable-only Adam state, merged LoRA export,
frozen-base fingerprints and actual decoding checks. Do not label this the
public multi-step TTT training recipe.

The pinned DeepSpec source005e03b81cec38b7da6399833d609ee89a2587f2 has a clean
working tree. Its Qwen3 EAGLE model forward combines post-fusion hidden[t] with
input token[t+1], then computes logits predicting token[t+2]. For an L-token
cached example, use features[:L-2], input_ids[1:L-1], and positions0..L-3.
The final15 output logits align with target logits at indicesL-16..L-2.
Labels are target-greedy IDs, matching the DFlash baseline's number of supervised
positions. This is one-step shifted teacher forcing; inherited embeddings,
head, norm and base weights stay frozen. Report distinct records and supervised
positions separately from feature regression's denser token objective.

Start from the completed dense512-example map after128 feature updates, SHA
92c5e2648d98d014d2a9968464c4ab73dba8b3905a7cce27da8d7f1b99048e33.
The first pilot uses64 existing cached examples and16 batch-four updates.
Four workers are connector-only CE, LoRA rank8, LoRA rank32, and a duplicate
rank32 seed. All start from the same mapper; LoRA freezes it. Use deterministic
algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8, lr2e-4, no weight decay, and
q_proj/v_proj LoRA with alpha twice its rank, matching the DFlash pilot.

Before fitting, compare teacher-forced prefix logits against the released
extend_draft_cache path on four records. Require exact initial-cache/logit
equivalence, preserved frozen weights, finite gradients, zero-LoRA identity,
merged export/reload equivalence, and duplicate fitted weights. The existing
first-update diagnostic also repeats the backward pass and records hashes.
Then compare fresh-request decoding on8 exposed prompts at128-token caps:
initial mapper, connector CE, both LoRA ranks, duplicate rank32, zero LoRA,
and runtime-local AR/native/source controls. Separate copied drafter instances
prevent candidate weight mutation from changing shared references. Row-level
adapter hashes bind the actual merged update used for decoding.

The pilot has a540-second process limit and10-minute Slurm allocation, four
GPUs total. It must pass before a512-example128-update calibration, bounded
rate checks and matched warm-time comparisons. Charge initial mapper fitting,
target-label preparation, optimizer work and export separately. More updates
reuse512 examples; no large-data pool resumes. The scientific matched-budget
comparison and quality evidence remain unexecuted.

Thirty local tests passed, covering shifted label alignment, mapper gradients,
malformed inputs, EAGLE backend behavior, LoRA and export/reload invariants, and
mapper campaign validation. Actual GPU cache equivalence, duplicate fitting and
longer generation are not established by these CPU tests. The pilot declaration
and pinned source/model/cache/checkpoint identities are in
reports/mapper-scaling-20260905/eagle3-adaptation-pilot/declaration.json.

Queued pilot27952 from immutable source844e066a3196ffab90250a27b90e5d641a3a97bd
at /home/aryama.murthy/relayspec-eagle-adaptation-844e066, afterany:27946.
Held submission was verified as four GPUs and ten minutes; the actual scratch
cache-index SHA matches the declaration. CUBLAS_WORKSPACE_CONFIG=:4096:8 is
explicit in submission. Ledger and collection use eagle3-adaptation-pilot/jobs.json.
This is a bounded infrastructure pilot; proper calibration, rate and time-budget
runs remain unsubmitted until it passes.
