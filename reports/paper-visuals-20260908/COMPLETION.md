# Completed graphical revision

Final PDF SHA256: `123ab1d05d2e8bff1a46dc68581c108ddb5060ab3fb43a19b013082425be8348`.

The anonymous manuscript has **9 main-text pages, 62 pages overall, 5 main figures, 28 figures overall, 3 main tables and 70 tables overall**. All50 citations remain resolved. The title and explicit four-contribution block are retained.

## Presentation changes

Thirteen new reproducible multi-panel figures were integrated: four main-paper figures (matched primary comparisons, data/capacity, warm adaptation budgets, family/native transfer) and nine appendix diagnostics (quality versus token agreement, block size, native compression, fitting loss versus throughput, task complexity, selective capture memory, verifier efficiency, per-request gain and observed lengths). Two older 14B trajectory figures were additionally given readable axis fonts, legends, markers and line styles at their existing size. Existing regularization, epoch, workload, composition and memory analyses remain.

The old primary throughput, adaptation-budget, rollout and family numerical tables moved to their appendix protocols. Main quality and heterogeneous public-baseline tables remain visible. Four plotting builders emit26PDF/PNG figure files and4plotted-value/hash registries. `make paper-visual-assets` regenerates the suite, and the full manuscript audit compares every output bytewise against a fresh build.

## Evidence and interpretation

All graphs use recorded experiments. No new GPU work was needed for this revision. Full ranges and negative results remain visible, including small MLP deficits, Llama's unresolved improvement, compression failures and the historical BF16 token disagreements. Cohorts, output caps, precision, references, fitting-seed limitations and selection exposure are specified in captions or adjacent protocols.

The added request-level transfer analysis shows122/128cross-family requests improve, with verification calls falling28,000→24,487(12.5%) for identical outputs. Llama calls are essentially unchanged16,066→16,094. These are descriptive recipe comparisons rather than data-count-only interventions. The older warm-budget caption's timing label was corrected from decoding to end-to-end tokens/s after checking its request-time aggregation.

## Verification

- Final manuscript audit: all16checks PASS. This includes official style, anonymity, 9-page boundary,50citations, allcore/research/family/newvisual assets, prompt audits, no duplicate prose, no LaTeX overfull boxes or undefined references, PDF parsing/embedded fonts, and current visual review.
- Relevant evidence tests: `tests/test_paper_evidence.py`, `tests/test_iclr_paper_assets.py`, `tests/test_ar_paper_evidence.py`: **10passed**. Matplotlib emitted existing dependency deprecation warnings.
- All62pages inspected incolor andgrayscale across three reviewers. Final pages34,42,43 were rechecked individually after minor caption/font corrections; allother59pages are pixel-identical to the fully reviewed candidate. See the threeVISUALreports and `final-render-comparison.json`.
- `git diff --check` passes.

## Reading copies

- `paper/iclr2027/relayspec_iclr2027.pdf`: canonical complete manuscript.
- `output/pdf/RelaySpec-ICLR2027-complete.pdf`: identical complete reading copy.
- `output/pdf/RelaySpec-ICLR2027-main9.pdf`: first-nine-page preview. References to omitted appendix pages are available in the complete PDF.

The graph/page index is in `paper/iclr2027/README.md`. This visual revision supersedes the older56-page reading draft; prior rebuild reports remain historical evidence rather than descriptions of the current PDF.
