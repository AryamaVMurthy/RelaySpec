# Repository cleanup — 5 September 2026

The project is named **RelaySpec**, with Python package name `relayspec` and
public repository `AryamaVMurthy/RelaySpec`.

## Structure changes

| Previous location | Current location |
|---|---|
| `paper/relayspec_v2/` | `paper/iclr2027/` |
| `scripts/build_relayspec_v2_figures.py` | `scripts/build_transfer_figures.py` |
| Superseded `docs/plans/` documents | `docs/archive/plans/` |
| Earlier research-report TeX and root bibliography | `docs/archive/` |
| Superseded report/DOCX builders | `scripts/archive/` |

The active plan stays in `docs/plans/`. The local repository directory is now
`RelaySpec`; the former `Latent-Recurrent` location is a compatibility symlink.
The old private GitHub repository is retained as history. The new public
repository starts with a clean snapshot rather than publishing private history.

## Removed or excluded

Removed about 89.8 MB of disposable review scratch data, generated document
exports, and caches. Generated research exports are recoverable from the local
pre-cleanup backup. Temporary dependency downloads and scratch renders can be
recreated. Git ignores environments, credentials, model weights, new run output,
and LaTeX intermediates. The current draft PDF and research evidence remain.

All 1,745 pre-existing report files were checked against the backup and were
unchanged after the move/cleanup phase. Scientific source snapshots and raw
experiment records were preserved, including unsuccessful experiments.
Historical references to old locations remain interpretable through this map.

## Standardization

Added package metadata, task-specific dependency extras, a refreshed lock,
Python formatting/lint settings, Makefile commands, GitHub code checks, a
separate strict manuscript audit, setup/reproduction documentation, and
upstream attribution notes. Plotting dependencies are pinned to the versions
used by the existing deterministic figure tests.

Formatting and unused-import cleanup do not address the scientific issues in
the September 5 review. Two existing manuscript tests still flag the stale
visual-review approval. Those checks remain available through `make test-all`
and `make audit-paper`; CPU CI does not claim submission readiness.
