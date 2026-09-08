# RelaySpec manuscript

Current version: **8 September 2026**, anonymous ICLR 2027 format, nine main-text pages and 62 total pages with statements, references and appendices. This is the current manuscript directory.

The paper centers on reusing frozen drafters through a learned linear interface. It distinguishes inexpensive calibration from the larger generated-rollout recipe, and separates historical BF16 Qwen measurements from the batch-invariant BF16 and FP32-target extensions. The main text includes the 128-question Llama/cross-tokenizer comparison at a 2,048-token cap, repeated native-throughput comparison, small-data confirmation, capacity findings, matched adaptation controls and public-runtime baselines. Detailed trajectories, negative results and implementation provenance remain in the appendix.

- `relayspec_iclr2027.tex`: manuscript source.
- `relayspec_iclr2027.pdf`: complete reading draft.
- `references.bib`: 50 cited sources, with current status inventory in `reports/paper-rebuild-20260908/`.
- `generated/` and `figures/`: reproducible result assets, with source fingerprints.

From the repository root, in the installed project environment:

```bash
make paper
make paper-family-assets
make paper-visual-assets
make audit-paper
```

For this shared-environment worktree, use `PYTHON=/home/aryamavmurthy/work/RelaySpec/.venv/bin/python` with the asset/audit targets. The Makefile puts this worktree's `src` first in `PYTHONPATH` so an editable installation from another checkout cannot silently supply stale evidence code.

`make paper-assets` rebuilds the primary AR/quality/source/native assets. Other specialized asset targets are listed in the Makefile. The manuscript audit regenerates registered core, scaling, autoresearch and new family-extension assets from their recorded raw inputs. The extension builder checks 128 unique requests per arm, direct token equality, hashes, numerical configurations, rollout source provenance, timing validity, both speculative repeats and paired request intervals.

Writer and reviewer rounds, current claim scope, source verification, visual review and final completion evidence are in `reports/paper-rebuild-20260908/`. The current visual-review record is tied to the PDF hash. Recompilation can change that hash and requires rechecking the rendered pages before updating its signoff. Presentation and artifact checks do not guarantee acceptance or resolve the explicitly stated experimental limitations.

## Visual revision, 8 September 2026

The paper now has **five main figures and 28 figures overall**. Thirteen newly generated multi-panel figures replace or extend the older presentation: four are in the main text and nine add appendix diagnostics. Complete numerical tables remain available.

| Analysis | Figure | PDF page |
| --- | --- | --- |
| Matched AR, native, source reuse and progress tradeoff | 2 | 5 |
| 16–32,768 records and downstream mapper capacity | 3 | 7 |
| Matched adaptation budgets | 4 | 8 |
| Family transfer and repeated native parity | 5 | 9 |
| Quality differences versus token agreement | 7 | 17 |
| Proposal block-size response | 9 | 19 |
| Native interface compression | 12 | 30 |
| Feature loss versus decoding throughput | 14 | 34 |
| Task difficulty and input-length subgroups | 18 | 41 |
| Selective feature capture and memory scaling | 24 | 57 |
| Verification efficiency | 25 | 60 |
| Per-request gains versus output length | 26 | 61 |
| Capped output and prompt-length distributions | 27 | 61 |

Existing workload, fitting trajectory, regularization, composition and source-memory figures are retained. New graphs use recorded runs rather than new GPU experiments. Four builders emit vector PDFs, PNG previews and complete plotted-value/input-hash registries. The manuscript audit rebuilds them and compares every output bytewise. Visual revision notes and final review are in `reports/paper-visuals-20260908/`.

## Contributions and competitor positioning, 8 September 2026

The introduction now presents four explicit contributions: portable frozen-drafter reuse through linear calibration, the small-data regime, the gap between feature fitting and decoding speed, and native-throughput/cross-family transfer. The application and empirical findings carry the novelty claim, with related linear-interface and frozen-adapter work credited.

Table 1 on page 2 directly compares RelaySpec, TriSpec, PARD and SD² by inherited model, adaptation and verification. Table 4 on page 8 reports the completed 128-question public-runtime comparison: RelaySpec 193.68 tokens/s (5.14× own AR), PARD 105.63 (3.34×), and frozen-drafter SD² 19.92 (1.60×). It marks the highest measured configuration values and states the runtime differences. TriSpec is a conceptual comparison. The main text has four tables, with 71 overall.

Section 6 is now solely Limitations. It identifies hidden-state access, checkpoint-specific calibration, preparation costs, serving conditions and the scope of greedy cross-tokenizer verification. Statistical and numerical qualifications remain adjacent to their results. Source checks, raw-record comparison review and the latest visual signoff are documented in `reports/paper-positioning-20260908/`.
