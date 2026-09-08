# Main-paper qualitative comparison

Authored 2026-09-08. The four-row table is `paper/iclr2027/generated/positioning_comparison_table.tex`. It is a conceptual comparison, not a new result registry. Existing bibliography keys are reused, and only this table and report are owned by this task.

## Suggested integration

Use a centered table with `\input{generated/positioning_comparison_table.tex}` and label `tab:positioning`. Suggested caption:

> Mechanisms of related reuse methods. TriSpec is compared conceptually; we do not benchmark it. PARD's reuse follows its original parallel-adaptation training. Our numerical SD² comparison tests its frozen-drafter variant, rather than its jointly trained default.

The main numerical comparison remains `tab:public-comparison`; this table must not be described as an experimental ranking or evidence that RelaySpec outperforms every method. OmniDraft is omitted for space and remains relevant to the related-work discussion.

## Cell-by-cell primary-source support

| Row | Inherited object | Adaptation | Execution / verification |
| --- | --- | --- | --- |
| RelaySpec | The manuscript's Method section (`sec:method`) defines the inherited feature-conditioned drafter and freezes its prediction network. | Equations `eq:interfaces` and `eq:loss` fit a linear input map; `eq:layer-context` gives the richer linear recipe. | “Draft with the new target's features” describes target verification and reuse of mapped states; the introduction specifies removal of the source transformer during generation. |
| TriSpec, `jiang2026trispec` | [§3.2](https://arxiv.org/html/2601.23180v1#S3.SS2) explicitly includes a pretrained EAGLE drafter with fixed weights in the adapter-only regime. This cell describes that regime, not the joint-training alternative. | The same section provides both adapter-only and joint training; §4.4 compares them. | §3.2's proxy pre-verification and confidence-margin test permit trusted proxy outputs; uncertain cases invoke the target. This is a different execution objective from removing a source transformer. |
| PARD, `an2026pard` | [Abstract and §3](https://arxiv.org/html/2504.18583v3) adapt an existing autoregressive draft model into a parallel predictor. | The abstract and introduction describe reuse within supported model families without separate target fitting. §3.2 supplies the original parallel-adaptation training. The [official repository](https://github.com/AMD-AGI/PARD) also states the original PARD model's target independence. | §§2.2–3.1 specify parallel candidate generation followed by target verification. The row is original PARD, not PARD-2. |
| SD², `berdoz2026sd2` | [§3.2](https://ojs.aaai.org/index.php/AAAI/article/view/40255/44216) uses a pretrained autoregressive drafter. | §3.3 jointly fine-tunes steering and drafter while freezing the verifier. §4.1 and Figure 6 also test a frozen-drafter variant and report gains from unfreezing. | §§3.1–3.2 preserve target verification and inject verifier-derived steering into drafter MLP layers. The table names both training regimes; our numerical baseline covers only the frozen variant. |

No claim of unique adapter-only training is made: TriSpec explicitly has that option, and SD² also studies frozen-drafter steering. The distinction presented is the inherited object, the fitted interface, and the resulting generation/verification arrangement. No universal losslessness claim is encoded in the table; the manuscript separately records numerical agreement limits.

## Layout and validation

The table uses `\small` (9-point in the current ICLR style), ragged-right paragraph columns, 3-point column padding, and no resizing. The fixed column widths total 5.17 inches; six internal 3-point gaps produce a nominal 5.419-inch table, below the requested 5.5-inch limit.

Standalone compilation with the actual ICLR style and resolved author-year citations passes without overfull boxes or unresolved references. TeX measures the complete input at 391.63573 points (5.419 inches); setup/end whitespace is suppressed explicitly. The 1,500-pixel page rendering at `tmp/pdfs/positioning-table/table.png` was visually inspected: all four rows and citations are legible, there is no clipping or collision, and the longest SD² citation wraps cleanly. The root agent handles final integrated-page review and pagination.
