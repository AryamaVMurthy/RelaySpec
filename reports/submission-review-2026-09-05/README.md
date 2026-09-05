# Review evidence

These are small local checks supporting the [submission review](../ICLR_SUBMISSION_REVIEW_2026-09-05.md), not new large-model benchmarks.

- `math-checks.json`: initial numerical counterexample to equivalence between raw weighted least squares and the post-normalization loss; also records architecture parameter counts.
- `checkpoint-check.json`: initial zero-training tiny-Qwen3 save/reload experiment. The current merge-and-save sequence leaves eight expected attention weights missing; an unloaded merged export reloads without missing weights and preserves scores.
- `reproduce_checks.py`: executable CPU reproduction. It extracts the current `save_merged_checkpoint` helper from the repository, uses a randomly initialized two-layer model, and downloads no model weights. Its objective example uses a different random toy construction than the initial check.
- `reproduced-checks.json`: successful output of that reproduction during the review. The post-normalization gradient remains nonzero at the raw least-squares solution. The checkpoint problem reproduces, while the unloaded export has zero missing/unexpected keys and zero output-score difference.
- `review-snapshot.json`: hashes identifying key reviewed files and the recorded repository test outcome.

Run instructions are in the reproduction script. The checkpoint check needs PEFT; the review installed it only in `tmp/submission-review-2026-09-05/deps`, leaving the project environment unchanged. Rendered PDF pages and verbose scratch logs are under the same temporary review directory.

The export finding is specific to the tested code and dependency versions. It does not establish which code created remote experiment artifacts. Auditing those actual artifacts remains a required next step.
