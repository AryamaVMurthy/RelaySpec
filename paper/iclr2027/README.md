# RelaySpec manuscript

This is the single current manuscript directory. The September 5 rewrite uses
recorded source-reuse comparisons with complete workload results.

- `relayspec_iclr2027.tex`: anonymous manuscript source.
- `references.bib`: bibliography.
- `generated/`: evidence-derived tables and the input-hash registry.
- `figures/`: six generated vector figures.
- `relayspec_iclr2027.pdf`: reviewed reading draft, nine main pages and 15 total.

From the repository root, `make paper` compiles the source and checked-in assets.
`make paper-assets` regenerates the assets through
`scripts/build_iclr_paper_assets.py` and `src/relayspec/paper_evidence.py`.
The builder validates the paired raw records and recorded task scores before
producing the current tables. Exploratory transfer builders remain separate.

The [rewrite plan](../../docs/plans/2026-09-05-relayspec-current-evidence-rewrite.md)
explains evidence selection, corrected claims and next experiments.
The [submission review](../../reports/ICLR_SUBMISSION_REVIEW_2026-09-05.md)
and research execution plan retain the outstanding scientific requirements.
Compilation and presentation checks do not establish submission readiness.

`make audit-paper` compiles and runs the strict artifact checks. A visual-review
record covers the current PDF in color and grayscale. Rebuilding can change its
hash, requiring inspection and an updated record before the audit passes.
LaTeX intermediate files are ignored by Git and rebuilt locally. Official
style files retain their original notices.
