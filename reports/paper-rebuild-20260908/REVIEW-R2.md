# Independent scientific review, round 2

Reviewed 8 September 2026 against the revised main text and extension appendices in `paper/iclr2027/relayspec_iclr2027.tex`, the new `family128_table.tex`, `rollout_transfer_table.tex`, `family_extension_evidence.json`, their builder, and the corresponding implementation/source artifacts. Line numbers below refer to the version inspected and may move during layout edits. No TeX was edited by this reviewer.

## Assessment

The scientific narrative is now substantially coherent. The revised title accommodates both short-text and generated-rollout calibration. Main prose distinguishes inherited model families, drafter architectures, numerical runtimes, record counts, development exposure, exact tokens and task accuracy. The stronger rollout result is correctly identified as custom Numina evaluation; it is no longer incorrectly used to imply an isolated loss improvement. The cross-family result retains the no-gain Llama control. The separate Discussion accurately limits what has been established.

The new main tables agree with the generated numerical evidence: family values use request throughput; the positive cross-family recipe gain is 14.315% with interval [11.548%,17.231%], while the Llama gain is 0.446% with interval [-0.626%,1.537%]. The rollout runs retain both measured native/relay values, conditional request intervals and the reused AR baseline. The generated-rollout layer/context equation matches the executed `Context.loss`, including equal layer/context weighting, raw layer inputs, frozen fusion/output normalization and the additive denominator epsilon. The explicit difference from the direct-context input normalization is a material improvement.

## Remaining fixes before sign-off

### R2-1. Correct the vocabulary-bridge implementation description

**Priority: high, factual prose/implementation mismatch.** Appendix around lines 2825–2828 says the intersection table accelerates translations “only when they agree with full chunk re-encoding.” This condition is not checked by the implementation. `src/relayspec/vocab_bridge.py:267–269` returns mapped token IDs whenever *every source token ID* appears in the table. Whole-chunk re-encoding is only performed when coverage is incomplete. The table builder checks singleton decoded-text equivalence, not canonical multi-token re-encoding.

Use: “Fully covered chunks use tokenwise vocabulary-intersection lookup; other chunks are decoded and re-encoded. Lookup preserves the table's singleton text matches but need not produce the target tokenizer's canonical chunk segmentation. The resulting candidate IDs are still verified by the target.” This is a documentation repair, not a reason to change the already measured runtime or invalidate the observed 128 exact sequences. Target-controlled verification does not require proposals to use canonical re-encoding.

### R2-2. Remove the unmeasured predictive claim

**Priority: medium, contribution overclaim.** Introduction line 82 calls the third contribution “Interface findings that predict deployment choices.” The listed evidence informs decisions after measurement; no held-out predictive policy or prospective winner forecast is established. Replace “predict” with “inform.” The same paragraph can retain its measured positive and negative outcomes.

### R2-3. Restore the nine-page main boundary

**Priority: high, user deliverable.** The compiled auxiliary file currently records `maintextend` on page 10; the log reports a 57-page total document. This violates the requested nine-page main text in the inspected build. Reduce redundant prose and move lower-priority display material to the appendix while preserving readable typography. This review does not certify the final rendering, which remains to be rechecked after layout revision.

### R2-4. Attach GPU counts to the richer recipe's costs

**Priority: medium, cost reproducibility.** The main and new appendix give stage times of 96.4, 52.0 and 14.4 minutes without identifying their different resource counts. The retained launch files show `generate.sbatch` requests two GPUs, `full_capture.sbatch` two GPUs and `full_train.sbatch` one GPU. The original short fit was timed on four GPUs. Add these counts at least in the appendix, and identify the times as stage wall time. Do not implicitly compare these stage times as equal-resource training runs. No new experiment is necessary.

## Useful minor improvements

- The setup sentence around lines 320–321 (“We score the same math outputs used for the AR timing comparison”) should say the **primary MATH-500 comparison**. The new extension outputs have exact-array evidence but were not separately answer-scored in the retained report. Existing later caveats make the intended scope clear, but precise setup wording avoids a global scoring claim.
- The new Llama table could include fitting records in its mapper labels, for example original 4,096 / selected 16,384 and original 4,096 / selected 8,192, if it fits comfortably. Currently the key larger-data takeaway depends on the appendix. The old/new recipe confound must remain in the caption.
- The word “richer” is accurate as a recipe distinction, but numerical claims should continue to specify the changed text source, supervision and work. Do not recast the result as richer mapping *capacity*: the folded interface has the same 52.43M parameter count.
- The new table builder reproduces counts, token-array matches, throughput and request intervals from raw records. Preserve a final source/configuration audit for the rollout precision and data identity as well; these facts are not all enforced by that builder alone. Hash validation of output records is not itself validation of the executed generation configuration.

## Scientific sign-off condition

After R2-1 through R2-4 and the setup-scoring wording are fixed, I see no remaining new submission-blocking scientific contradiction in the revised main narrative or extension equations. The previously identified unrun evidence remains limited fitting-seed replication, runtime-confounded public comparisons, one heterogeneous pair, and single-request throughput. These limits are now stated rather than concealed. This is a defensible paper presentation, not a guarantee of ICLR acceptance or a claim that every research question is resolved.

Final sign-off still requires inspecting the corrected text and the final nine-page rendering, checking assets against the final source, and retaining a clear division between format QA and scientific review.
