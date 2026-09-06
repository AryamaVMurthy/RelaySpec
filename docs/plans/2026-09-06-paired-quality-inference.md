# Paired quality inference before untouched confirmation

The percentile bootstrap collapses to zero width when every observed candidate
minus AR correctness difference is zero. This follows directly because every
resample then also has mean zero. It cannot exclude unobserved rare adverse
discordances. Keep existing development bootstrap summaries descriptive.

Before model selection or confirmation generation, use a conservative two-sided
interval for the primary quality criterion. For each IID request pair define
beneficial discordance as candidate correct and AR wrong, and adverse discordance
as candidate wrong and AR correct. Their probabilities are p_plus and p_minus.
The accuracy difference is p_plus minus p_minus. Each count has a binomial
marginal, although the counts are dependent.

Compute exact97.5% Clopper-Pearson intervals for each probability. By the union
bound, both intervals cover simultaneously with probability at least95%.
Subtract endpoints to obtain [L_plus-U_minus, U_plus-L_minus]. This is an
explicit conservative construction, not a claim that the two counts are
independent or that the interval is optimal. The proof applies under IID
request pairs and fixed model/scorer selection, and does not establish global
workload generalization or absence of dataset contamination.

The256 primary requests, one-percentage-point margin and reserve policy are
unchanged. Zero discordances at256 requests give a nonzero-width interval and
cannot establish this margin. Inconclusive is an allowed result. Do not add
reserve examples after viewing the primary outcome. Preserve the prior
protocol under its recorded SHA and state this pre-generation amendment.

Implementation is src/relayspec/paired_accuracy.py using SciPy1.16.1. Tests
cover the analytic zero-discordance boundary, pairing and reversal, invalid
outcomes, and enumerated coverage on a small multinomial probability grid.
The grid test checks implementation behavior. The union-bound argument above
provides the general coverage justification.

## Verified source log

- SciPy community, *BinomTestResult.proportion_ci*, version1.16.1, official
  software documentation. Verified6September2026. The `exact` option uses
  Clopper-Pearson limits. This supports the marginal interval implementation.
  https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html
- NIST/SEMATECH, *7.4.7.3. Bonferroni's method*, official technical handbook.
  Verified6September2026. Its general event inequality supports simultaneous
  coverage without an independence requirement. The paired-discordance
  construction above is our application of that inequality, not a procedure
  attributed verbatim to this ANOVA-oriented handbook page.
  https://itl.nist.gov/div898/handbook/prc/section4/prc473.htm
