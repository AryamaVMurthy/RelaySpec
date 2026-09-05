# Qwen3-8B conservative joint-objective MATH-500 ablation

## Protocol

- Full 500-prompt MATH-500, Qwen3 non-thinking greedy decoding, 2,048-token
  cap, batch one, and DFlash block size 16.
- Qwen3-8B target and the released Qwen3-4B DFlash proposer are frozen.
- The relay starts from the selected feature-only checkpoint and is refined
  for 1,024 steps with feature MSE, cosine distance, and a detached-teacher
  proposal KL term of weight 0.1.
- Generation job `25514` completed in 1:04:11 on exactly four RTX 6000 Ada
  GPUs; scorer job `25516` completed successfully.

## Result

| Method | End-to-end tok/s | vs source reuse | Mean accepted | Official accuracy | Source=method |
|---|---:|---:|---:|---:|---:|
| Native AR | 44.480 | 0.305x | 1.000 | 74.8% | 122/500 |
| Source-trunk reuse | 145.919 | 1.000x | 7.760 | 74.2% | 500/500 |
| Target-specific DFlash-8B | 240.755 | 1.650x | 8.014 | 74.2% | 500/500 |
| Conservative joint relay | **218.750** | **1.499x [1.491, 1.508]** | 7.006 | 74.2% | 500/500 |

The joint relay is exact relative to source reuse on all 500 prompts and has
the same official accuracy. Its direct same-prompt comparison with the
feature-only relay is:

| Objective | End-to-end tok/s | Mean accepted |
|---|---:|---:|
| Feature-only | 218.642 | 6.992 |
| Conservative joint | 218.750 | 7.006 |

The conservative/feature ratio is **1.0005x**, with a 10,000-sample paired
bootstrap 95% CI of **[0.9981, 1.0028]**. The objectives also produce the same
token sequence on 500/500 prompts. The 0.049% point gain is not statistically
resolved, so the simpler feature-only objective remains the primary method;
the joint loss is retained as the objective ablation.

## Artifact checks

- Four rank files contain exactly 500 rows each: 2,000 unique method/prompt
  records in total.
- `benchmark-paper-summary.json` contains the official score, request-level
  metrics, prompt-paired intervals, and component profiles.
- `objective-comparison.json` records the direct feature/joint comparison.
- `allocation.jsonl`, `gpu.csv`, `config.yaml`, manifest/hash, model/code
  revisions, raw JSONL, scored JSONL, and source hashes are included here.
