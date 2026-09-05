# Development

The project uses a `src/` package layout, `pyproject.toml` for metadata,
`uv.lock` for dependency resolution, Ruff for Python checks/formatting, and
pytest for tests. Run commands from the repository root.

| Command | Purpose |
|---|---|
| `uv sync --locked` | Install runtime and development dependencies |
| `make format` | Apply consistent Python formatting |
| `make check` | Lint, formatting check, and CPU code/evidence tests |
| `make test-all` | All tests, including built-manuscript checks |
| `make paper` | Compile the draft using checked-in figures |
| `make paper-assets` | Regenerate tables/figures; review differences before committing |
| `make audit-paper` | Compile and run the strict manuscript-artifact audit |
| `make clean` | Delete Python/LaTeX intermediates; retain evidence and PDF |

## CPU and GPU environments

The lock includes the normal PyTorch distribution. A CPU-only development
environment can instead install PyTorch from its official CPU wheel index:

```bash
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e '.[paper]' pytest ruff sympy
make check
```

This is the GitHub Actions smoke-test environment, not the locked GPU
experiment environment. Model benchmarks must record actual PyTorch, CUDA,
Transformers, and upstream drafter versions. Historical EAGLE-3 runs used a
separate DeepSpec Python overlay; inspect the launchers before assuming the
base package reproduces that stack.

## Manuscript checks

Install `pdflatex`, `bibtex`, Poppler (`pdfinfo`, `pdftotext`, `pdffonts`), and
Ghostscript (`gs`). Ubuntu packages include `texlive-latex-extra`,
`texlive-fonts-recommended`, `texlive-science`, `poppler-utils`, and `ghostscript`.

Tests marked `manuscript` inspect a compiled paper, including auxiliary files
and an explicit visual-review record. They are separated from CPU CI because
they need external tools and a recorded visual review. They remain in `make test-all` and
the optional **Manuscript audit** GitHub workflow.

The current manuscript has a page-by-page color and grayscale review recorded
against its exact PDF hash in `reports/ICLR_VISUAL_REVIEW.json`. Recompiling can
change that hash, so a previous visual review must not be copied automatically.
Inspect the rendered pages and update the record before running the final audit.
The artifact audit is separate from the scientific completion requirements in
the submission review and research execution plan.

Archived document builders in `scripts/archive/` are retained for provenance;
they are outside the supported build and active lint/format target.
