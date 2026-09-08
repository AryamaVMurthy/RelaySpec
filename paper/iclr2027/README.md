# RelaySpec manuscript

Current version: **8 September 2026**, anonymous ICLR 2027 format, nine main-text pages and 56 total pages with statements, references and appendices. This is the current manuscript directory.

The paper centers on reusing frozen drafters through a learned linear interface. It distinguishes inexpensive calibration from the larger generated-rollout recipe, and separates historical BF16 Qwen measurements from the batch-invariant BF16 and FP32-target extensions. The main text includes the 128-question Llama/cross-tokenizer comparison at a 2,048-token cap, repeated native-throughput comparison, small-data confirmation, capacity findings, matched adaptation controls and public-runtime baselines. Detailed trajectories, negative results and implementation provenance remain in the appendix.

- `relayspec_iclr2027.tex`: manuscript source.
- `relayspec_iclr2027.pdf`: complete reading draft.
- `references.bib`: 50 cited sources, with current status inventory in `reports/paper-rebuild-20260908/`.
- `generated/` and `figures/`: reproducible result assets, with source fingerprints.

From the repository root, in the installed project environment:

```bash
make paper
make paper-family-assets
make audit-paper
```

For this shared-environment worktree, use `PYTHON=/home/aryamavmurthy/work/RelaySpec/.venv/bin/python` with the asset/audit targets. The Makefile puts this worktree's `src` first in `PYTHONPATH` so an editable installation from another checkout cannot silently supply stale evidence code.

`make paper-assets` rebuilds the primary AR/quality/source/native assets. Other specialized asset targets are listed in the Makefile. The manuscript audit regenerates registered core, scaling, autoresearch and new family-extension assets from their recorded raw inputs. The extension builder checks 128 unique requests per arm, direct token equality, hashes, numerical configurations, rollout source provenance, timing validity, both speculative repeats and paired request intervals.

Writer and reviewer rounds, current claim scope, source verification, visual review and final completion evidence are in `reports/paper-rebuild-20260908/`. The current visual-review record is tied to the PDF hash. Recompilation can change that hash and requires rechecking the rendered pages before updating its signoff. Presentation and artifact checks do not guarantee acceptance or resolve the explicitly stated experimental limitations.
