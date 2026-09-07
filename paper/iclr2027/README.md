# RelaySpec manuscript

This is the single current manuscript directory. The September 5 rewrite uses
paired AR throughput as the primary comparison, native-drafter throughput
retention as the reuse reference, and source reuse as a secondary control.

- `relayspec_iclr2027.tex`: anonymous manuscript source.
- `references.bib`: bibliography.
- `generated/`: evidence-derived tables and the input-hash registry.
- `figures/`: generated vector figures, including the method and decoding flows.
- `relayspec_iclr2027.pdf`: current anonymous reading draft, nine main pages.

From the repository root, `make paper` compiles the source and checked-in assets.
`make paper-assets` regenerates the assets through
`scripts/build_iclr_paper_assets.py`, `src/relayspec/paper_evidence.py` and
`src/relayspec/ar_paper_evidence.py`.
The builder validates the paired raw records and recorded task scores before
producing the current tables. Exploratory transfer builders remain separate.

The [current execution record](../../reports/ar-revision-20260905/EXECUTION.md)
and [24-hour plan](../../docs/plans/2026-09-05-relayspec-next-24-hours.md)
explain evidence selection, corrected comparisons and next experiments.
The [submission review](../../reports/ICLR_SUBMISSION_REVIEW_2026-09-05.md)
and research execution plan retain the outstanding scientific requirements.
Compilation and presentation checks do not establish submission readiness.

`make audit-paper` compiles and runs the strict artifact checks. A visual-review
record covers the current PDF in color and grayscale. Rebuilding can change its
hash, requiring inspection and an updated record before the audit passes.
LaTeX intermediate files are ignored by Git and rebuilt locally. Official
style files retain their original notices.

The current draft uses the unmodified official ICLR 2027 review style and
bibliography style. It has nine main-text pages and 53 pages including references
and appendices. The manuscript audit resolves 47 cited sources. The September 5
citation-status report is historical, while later additions are recorded in
`reports/citation-expansion-20260905.json`. Current evidence includes native
interface confirmation in both drafter families, code-capacity controls and
full-answer 14B development comparisons. Sampling results use matched DFlash proposal policies, with the historical
policy difference and the full-projection reproduction control explained.

The scaling branch adds completed fixed-work data and continuous-fitting curves
and AR-paired EAGLE-3 code quality to the appendix. Regenerate these assets with
`make paper-scaling-assets SCALING_RAW_ROOT=/path/to/collected/repository`.
The builder verifies original benchmark/scorer file hashes. The additional
registry records the raw root, and the manuscript auditor regenerates these
assets too. New capacity, regularization and multi-epoch studies remain in
progress and are not represented as completed results in the paper.

