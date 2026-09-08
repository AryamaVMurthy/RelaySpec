# Writer review, revision 2

Reviewed the actual revised source and the compiled PDF on 2026-09-08. The source already integrates the new NuminaMath native-parity result and the 128-question family results correctly. The compiled `maintextend` label is **page 10**, so the requested nine-page main paper is not yet achieved. The highest-value remaining work is structural compression, not removing the new results.

## Overall assessment

The central message is now persuasive and properly scoped: frozen speculative prediction is reusable through a learned interface; cheap calibration works well; richer calibration can reach native speed; transferring families exposes where additional fitting helps. The direct-context/layer-plus-context distinction is present, and the draft no longer silently generalizes BF16 numerical agreement or Qwen vocabulary compatibility.

The main weakness is that the **result order still feels like the new experiments were inserted into the old paper**. Sections 5.7 and 5.8 introduce richer calibration and family transfers before Section 5.9 establishes the small-data and capacity findings. The richer section starts with “The small-data regime is not the only useful operating point,” but the reader has not yet seen that regime's evidence. A simple reorder will make the paper feel rebuilt rather than extended.

## Recommended final result order

1. Primary acceleration, native retention/source removal, task-quality and numerical agreement, then breadth. This establishes that reuse has practical value.
2. Calibration/data/capacity and frozen small-data confirmation, including the compact matched CE/LoRA comparison. This explains why the default is inexpensive and why simplicity is supported by alternatives.
3. Richer generated-rollout calibration. This tests whether the original native gap is a ceiling and establishes a different operating point.
4. Llama and cross-tokenizer transfer. This tests model breadth and reveals different returns to extra fitting.
5. Public-runtime alternatives, discussion/limitations, conclusion.

Move the whole current ablation block ahead of the richer-calibration subsection and rename it **“Calibration cost and interface capacity.”** Consider renaming its child heading to **“Data, capacity and frozen confirmation”**. Its findings are part of the paper's contribution, not secondary ablations. Public-runtime alternatives should be a peer subsection, not an unnumbered paragraph nested inside the adaptation-budget subsection: it answers a different question with a different comparison contract.

The data curve and frozen confirmation table should appear consecutively. The capacity paragraph currently intervenes. Put it after the confirmation table, immediately before “Better feature fitting need not mean faster decoding.” This gives a clean sequence of exploration → frozen confirmation → explanation of capacity and fitting behavior.

## Precise compression plan for page 9

These cuts preserve all substantive evidence and should save roughly one page together. Compile after the prose/heading changes before shrinking figures. Do not alter the official style or squeeze tables to unreadable size.

### 1. Repair the three-line title

The explicit line break currently yields **three lines** in the PDF: “RelaySpec: Reusing Frozen Speculative” / “Drafters” / “through Learned Feature Interfaces.” Replace it with a shorter, two-line title:

```tex
\title{RelaySpec: Reusing Frozen Drafters\\through Learned Linear Interfaces}
```

This is also more precise about the contribution than the generic “Feature Interfaces.”

### 2. Fold the short preliminaries into method/setup

The separate Preliminaries section and its two subsection headings cost substantial space for material repeated in the method. Keep the two metric definitions, but move them to one `\paragraph{Metrics.}` in setup. Put the greedy-verifier definition in the inference subsection, which already explains the same procedure. The existing `sec:preliminaries` label can attach to this method introduction if other text refers to it.

Replace the current metric prose before/after Equation 1 with:

> We measure end-to-end throughput as total generated tokens divided by summed request time, including prompt processing. Let $T_A$, $T_R$ and $T_N$ denote throughput for AR, RelaySpec and native drafting. [Retain Equation 1.] We also report summed-time ratios when output lengths differ. Accepted progress counts committed tokens per verification cycle, including the target-supplied token (Appendix E).

This removes a duplicated “Throughput includes prompt processing” sentence and details already covered in source removal/acceptance. The method still must explicitly state longest matching prefix plus target next token.

### 3. Remove table narration from the primary acceleration paragraph

Replace the full paragraph starting “Table 1 reports absolute end-to-end throughput...” through “Each paired interval is above one” with:

> RelaySpec achieves 2.35–5.11 times paired AR throughput on MATH-500, with every interval above one (Table 1). The native and source-reuse controls below measure how much inherited drafting performance is retained and whether source removal pays for the changed proposals.

Table 1 already contains all eight absolute numbers. This saves approximately 45–55 words and makes the paragraph explain the comparison.

### 4. Merge fitting/memory into source removal

The separate “Fitting work and deployment memory” heading repeats the source-removal topic. Keep its hard numbers, but attach a compact paragraph to “Native retention and source removal”:

> Original Qwen maps contain 52.43–65.54M weights and take 85.6–118.5 seconds to fit on four RTX 6000 Ada GPUs after loading, including feature computation. Separate EAGLE-3 processes reduce peak allocated memory by 7.80 GiB at 8B and 7.63 GiB at 14B. Required source embedding/head tensors remain resident; the removed component is its transformer (Appendix G).

The complete before/after memory values, loading/export details and alternative numerical runtimes remain in the appendix and richer-calibration subsection. Crucially, “Original Qwen maps” replaces the now-ambiguous “selected maps”: the measured sub-two-minute fitting cost does not apply to all new recipes.

### 5. Shorten repeated self-positioning in related work

Keep the substantial closest-prior-work coverage. Trim repetition of this paper's experiments:

- TriSpec paragraph: delete “Adapter-only tuning is therefore not our novelty claim.” The comparison already credits adapter-only tuning explicitly. Replace its final two sentences with: “RelaySpec removes the source transformer and retargets the inherited drafter while retaining target-controlled acceptance; we study the associated calibration and deployment tradeoffs.”
- PARD paragraph: delete “Thus reuse across targets is established prior work.” The preceding sentence already states exactly that fact. End the paragraph after the sentence specifying own matched AR controls; the final two lines re-explain the baseline caveat given in its main table caption and limitations.
- Distillation paragraph: replace the four sentences beginning “Our connector-CE...” with: “Matched-budget connector-CE and rank-32 LoRA controls test this distinction empirically (Table 6).” Keep the fuller result and limitations in the actual adaptation subsection.
- Frozen interfaces paragraph: replace the last three sentences with: “Stitching need not imply equivalent information (Smith et al., 2025), motivating evaluation of decoded speed and quality alongside feature error.” The capacity experiment and frozen confirmation are already introduced twice elsewhere.
- Execution paragraph: after the examples/citations, finish with: “RelaySpec changes the feature source while retaining the released proposal procedure.” The following five lines restate scope and own-AR controls already established.

These edits save roughly 140–180 words without removing relevant literature or weakening credit to prior work.

### 6. Trim redundant qualification without removing its substance

- In “Better feature fitting...”, all mappers' 105/128 scores and unresolved one-point margin repeat earlier data-sweep and quality/confirmation discussion. Retain the scores once in the data paragraph/table; remove the repeated two lines here.
- End that capacity paragraph after the sentence stating dense leads both completed 8B long-output capacity comparisons. The next two sentences repeat the introduction's premise and the appendix existence.
- In the adaptation paragraph, “Every paired 95% interval lies below parity” plus “supports feature regression as the best tested adaptation...” repeats the first sentence and table. Retain the intervals and the short-screen limitation; delete the intervening restatement.
- The richer section can end its first paragraph with “This establishes native-level throughput for this fitted recipe and runtime.” Delete “rather than an upper bound imposed by reusing a 4B drafter,” which introduces an unnecessary implied theoretical ceiling.
- In the family section, retain the cohort-exposure statement and numerical contract. The sentence “We do not pool its speeds with the BF16 Qwen suite” can move to setup once all separately identified runtimes are made clear; avoid repeating it in several sections.

These edits save approximately another 80–110 words. The main-text limitations should remain.

## Small wording changes for scientific precision and readability

1. Contribution 3, **“Interface findings that predict deployment choices,”** overstates prospectivity. The work mostly measures and explains choices. Prefer **“Evidence for interface design”** or **“Benefits and boundaries of interface adaptation.”**
2. Related work says “RelaySpec holds that behavior fixed.” The drafter's **weights** remain fixed; its proposals and acceptance behavior change with the conditioning. Replace with “RelaySpec holds the drafter's prediction weights fixed and regresses its input context.”
3. The intro's “preserving ... proposal behavior” can imply identical proposals. “Retaining useful proposal behavior” expresses the intended claim accurately.
4. The richer objective's `\mathcal L_context` definition is compact but enough if the appendix details are retained. Add “with record-wise averaging” only if needed to tie `n` in the equation unambiguously to a record; avoid implying uniform global sampled-position weighting.
5. In “Beyond Qwen scaling,” prefer **“Transfer across model families and tokenizers”**. It directly names the question, and the smaller Llama target is not a conventional scaling point.
6. Caption Table 4 should say **“Original and selected dense maps”** if the actual generated table doesn't identify model class. Otherwise readers could confuse “selected” with the layer-plus-context recipe introduced immediately before it.
7. Discussion currently contrasts “BF16 output differences and exact FP32-target matches,” omitting the positive batch-invariant BF16 experiment. Replace with “Historical BF16 output differences and exact matches in the batch-invariant BF16 and FP32-target studies must be read under their respective numerical references.”
8. The main table caption “Progress R” has no plain-language payoff unless a reader knows the implementation. “Relay progress counts committed tokens per verification cycle” is clearer and avoids an extra abbreviation definition.

## Appendix figure placement

The relocated headline AR figure is acceptable after the complete AR details table in the implementation/measurement appendix: it visually summarizes the same suite and retains its original caption. Main text already has the primary numerical table, so moving this redundant figure is a good trade.

The relocated `numina_capacity.pdf` is **currently misplaced** at the end of “Complete workload results,” immediately after the EAGLE-3 code-quality table and immediately before “Fitting and interface studies.” Move it inside the fitting section beside the Numina capacity discussion/table, ideally near `tab:numina-capacity` or its first subsection. It should not look like a code-benchmark result. The main reference “Appendix Figure X” is otherwise clear.

The appendix's opening contents sentence should add the two new extension appendices. At present it lists implementation, workload, fitting, acceptance, data, and fitting/memory costs, but omits the large newly added operating-point and family-transfer evidence. A final clause is enough: “and the generated-rollout and cross-family extensions (Appendices J–K).” Use labels rather than hardcoded letters.

The protocol table at the start of the appendix still says **“Evaluation protocol. Main fitting choices...”** with global BF16/block/data settings. Rename it **“Primary Qwen evaluation protocol”** and add a one-sentence pointer to the separate extension protocols. The surrounding “Released models” paragraph should likewise begin “The primary suite's source...” rather than implying every reported source is Qwen3-4B.

## Revision disposition

The scientific story is strong enough for a coherent completed rewrite once the result order and ten-page overflow are fixed. The new positive experiments should remain in the main paper. The remaining cuts should remove duplication and excess heading space; they should not delete numerical-reference, data-exposure, cost, or quality qualifications needed to interpret the headline results.
