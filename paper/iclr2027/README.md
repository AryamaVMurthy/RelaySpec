# RelaySpec manuscript

This is the single current manuscript directory.

- `relayspec_iclr2027.tex`: manuscript source.
- `references.bib`: bibliography.
- `generated/`: tables/macros produced from recorded evidence.
- `figures/`: generated vector figures.
- `relayspec_iclr2027.pdf`: current reading draft.

From the repository root, `make paper` compiles existing source/assets;
`make paper-assets` regenerates assets. Builders are
`scripts/build_iclr_paper_assets.py` and `scripts/build_transfer_figures.py`.

The manuscript remains under scientific revision. Read
[the submission review](../../reports/ICLR_SUBMISSION_REVIEW_2026-09-05.md).
Compilation, formatting, and visual inspection do not validate experiments.

`make audit-paper` runs strict submission checks. The visual-review record
refers to an older PDF and must be renewed through actual inspection.
LaTeX intermediate files are ignored by Git and rebuilt locally. Official
style files retain their original notices.
