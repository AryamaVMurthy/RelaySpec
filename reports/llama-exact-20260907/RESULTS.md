# Llama exact-output and speed pass

The established normalized-linear mapper with FP32 target execution is the
best validated recipe in this short comparison. Both it and the ZIP
20-epoch mapper match the corresponding plain FP32 AR decoder on all 16
tested prompts in each family (64 paired comparisons over four recipes).
This resolves the observed agreement failures on the tested requests while
retaining a decoding speedup. It is not a universal floating-point guarantee.

## Pilot: eight MATH prompts, cap 512 output tokens

| Transfer | Plain AR TPS | Old mapper TPS | ZIP20 TPS | Exact agreement per mapper |
|---|---:|---:|---:|---:|
| Llama-3.1-8B drafter to Llama-3.2-3B | 46.8 | 107.7 | 91.5 | 8/8 |
| Qwen3-4B drafter to Llama-3.1-8B | 22.3 | 49.9 | 48.3 | 8/8 |

## Confirmation: eight disjoint prompts, cap 1,024 output tokens

| Transfer | Plain AR TPS | Old mapper TPS | ZIP20 TPS | Exact agreement per mapper |
|---|---:|---:|---:|---:|
| Llama-3.1-8B drafter to Llama-3.2-3B | 46.3 | 91.4 | 77.1 | 8/8 |
| Qwen3-4B drafter to Llama-3.1-8B | 22.0 | 61.7 | 60.7 | 8/8 |

The old mapper's confirmation decoding speedups are about 1.98x and 2.80x.
Exact summed request times, including prefill, are in `measurements.json`.
TPS values are aggregate tokens divided by aggregate decode time, not
arithmetic means of per-request speeds. Each method runs on one L40S GPU;
four independent lanes run concurrently. The different caps and prompt
sets are reported separately, not combined into a single benchmark speed.

## Interpretation and limitations

- Target, source interface, drafter, and mapper were FP32; TF32 was disabled.
  No acceptance rule, tokenizer bridge, or cache-cropping code was changed.
  The source transformer was unloaded after retaining its embedding/head.
- Precision is part of the baseline contract: FP32 speculation was compared
  with plain FP32 AR. This does not preserve the old BF16 AR output stream.
  On the pilot, FP32 AR matched the original BF16 AR on only 4/8 Llama and
  2/8 cross-family prompts. Both numerical references must be labelled.
- Agreement means complete output-token sequence equality, not MATH answer
  correctness. Both arms obey the same EOS rule and token cap. It does not
  imply uncapped answers have been tested.
- Old and ZIP checkpoints use different training budgets; this selects an
  available practical recipe, not an isolated loss-function ablation.
- The old mapper consistently leads on Llama. The cross-family differences
  between old and ZIP20 are small and do not establish statistical superiority.
- Main jobs: 28882 (pilot), 28885 (confirmation). Raw rows, configs, and
  numerical agreement gate are preserved here. No retraining was required.

A separate short mixed-precision screen (job 28888) keeps the target and
mapper FP32 but makes the frozen drafter/embedding/head BF16. Its smaller
four-prompt-per-family scope cannot replace the larger confirmation above.

Mixed screen results (four requests per family, cap 512):
- llama: AR 46.80 TPS, relay 132.13 TPS, 4/4 exact matches.
- cross: AR 22.26 TPS, relay 50.17 TPS, 4/4 exact matches.

Jobs 28882, 28885, and 28888 all completed with exit code 0; no GPU jobs remain. Total pass finished within 15 minutes. The full-FP32 configuration is the best-validated recommendation; the mixed result is a smaller performance pilot.
