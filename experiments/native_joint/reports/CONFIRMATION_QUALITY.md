# Confirmation answer and code audit

Math uses the vendored Qwen2.5-Math extraction/grading rules. HumanEval uses its supplied standard tests in an isolated sandbox. All scores use one deterministic completion per request; repeated identical outputs are not extra samples. Failures and extraction details are retained in quality-scored.json. Dialogue has no ground-truth score and is not claimed equivalent.

| Method | GSM8K correct /32 | MATH500 correct /32 | HumanEval passed /32 | Dialogue capped /32 |
|---|---:|---:|---:|---:|
| native | 28 | 30 | 31 | 0 |
| compact_linear | 28 | 30 | 31 | 0 |
| candidate | 27 | 29 | 31 | 0 |
| reference | 29 | 28 | 30 | 0 |
| ar | 29 | 30 | 30 | 0 |

These small-set scores do not prove distributional equivalence or universal accuracy preservation. BF16 exact AR/native agreements are reported separately in CONFIRMATION.md. Review logged extraction/test failures before attributing a quality difference to a model.
