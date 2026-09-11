# Ongoing AUF study: preliminary evidence report

This is the separate new manuscript directory. The old RelaySpec manuscript
is preserved. This draft contains only completed development results; it is
not submission-ready and does not imply that queued experiments have run.

From the RelaySpec-AUF worktree:

```bash
/home/aryamavmurthy/work/RelaySpec/.venv/bin/python -m experiments.auf_vllm.build_paper_assets --reports experiments/auf_vllm/reports --out paper/auf_iclr
cd paper/auf_iclr
TEXINPUTS=.:../iclr2027: pdflatex -interaction=nonstopmode -halt-on-error main.tex
TEXINPUTS=.:../iclr2027: pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Keep the current directory first in TEXINPUTS so the new generated tables
take precedence over similarly named assets in the template directory.

The builder validates paired tokenized prompts, 128 unique completed requests,
2,048-token caps, timing validity, complete output equality, main fitting
contracts, and the deployed LoRA projection checks. `claim_evidence.json`
records source hashes and incomplete requirements. The comparisons are
single-seed/single-timing development evidence until additional reports are
explicitly collected and the builder is extended.

Experiment protocols, Slurm dependencies and progress are in
`docs/plans/2026-09-11-auf-vllm-study.md` and
`experiments/auf_vllm/STATUS.md`. Four GPUs is the total maximum; queued
arrays do not constitute completed evidence.
