# Manuscript evidence review before final rewrite

Current manuscript inspected: paper/iclr2027/relayspec_iclr2027.tex, committed version before the post-window rewrite. This is a review, not a completed paper update.

## Main-text changes required

1. Preserve the original fixed-checkpoint MATH-500 and breadth results as one protocol. Their 4,096-record/1,024-update recipe must not be silently replaced by the separate Numina 512/2,048-record/8,192-update fits.
2. Promote controlled fixed-update data scaling and the continuous-training trajectory. The current main-text sweep jointly changes data and optimization; it cannot establish data efficiency by itself.
3. Promote the complete matched linear/MLP capacity study, using validation error as a fitting metric and measured decoding as a separate outcome. Explain the factorization rank ceiling and matched parameter counts.
4. Add a compact two-family long-output comparison: DFlash dense N512 retains 97.8% and EAGLE dense N512 retains 98.3% of their respective dense N2048 throughput. Preserve development exposure, output caps, conditional bootstrap intervals and single-fit scope.
5. Give the 14B replication a main-text paragraph with its counterexample: the DFlash MLP4096 N2048 improves training loss but worsens validation loss and short decoding; EAGLE exhibits late optimization degradation. The lower-rate pilot is not a resolution because proper endpoint fits were not run.
6. Include the matched warm-optimizer-budget DFlash feature/CE/LoRA comparison in the main evidence. Separate warm optimizer time from feature preparation, model loading and export. Keep the 16-request/256-token scope explicit.
7. Keep task difficulty and length findings descriptive. The longest-input subgroup has four questions; do not infer a general complexity law.
8. Shorten redundant method-capability tables and move inherited drafter recipe counts to the appendix. These are provenance/context, not a matched adaptation-cost experiment.
9. Update the abstract, introduction, contributions and conclusion together so the main story matches the new controlled evidence while retaining the original broader deployment results.

## Claims that require repair

The existing reports/claim-evidence-map.md uses “quality is unchanged,” “confirmatory,” and broad “generalizes” language that exceeds the current paper scope. Replace those with observed paired outcomes and their finite-sample limits. No claim of tight accuracy non-inferiority, bitwise BF16 equivalence, untouched confirmation, universal minimal data, optimal architecture, total-compute superiority, or production serving throughput is justified by the new audits.

## Presentation and verification after rewrite

- Keep the submission main text within nine pages; support essential conclusions there and place full grids in the appendix.
- Wire the completed EAGLE quality asset exporter into the reproducible paper build and audit.
- Rebuild all affected tables/figures from audited registries; check citations, references, layout, fonts, anonymity and the actual main-text boundary.
- Visually inspect every final PDF page in color and grayscale and update the review record with the final PDF hash. The current visual record is stale and must not be treated as passing.
- Preserve a clear list of unrun work: composition fits/quality, full 14B long-answer capacity comparison, proper lower-rate EAGLE14 fits, generalized EAGLE adaptation, untouched confirmation and autoresearch.

## Additional interpretation checks

- The historical fitting/evaluation lexical check finds 47 MATH questions above 0.80 token-set overlap and four above 0.95. Zero exact overlap does not establish template independence. Mention this limitation in the main discussion and distinguish the separate Numina filtering protocol.
- SD-square and PARD use public runtimes with different precision, attention and Transformers settings. Their own AR-relative comparisons are useful baselines; absolute cross-runtime tokens/s do not isolate the algorithm. Preserve both baseline results rather than using only the favorable comparison.
- The eight numerical-divergence traces diagnose selected BF16 cases. They do not prove every output difference has the same cause, or that approximate arithmetic is lossless.
- The old 468-question complement was computed from exposed saved runs. It is not newly collected untouched confirmation.
- Current appendix prose discusses a final one-point non-inferiority protocol that was not executed. The final manuscript should phrase this as an unestablished quality claim, not as an outstanding submission promise or a completed test.
