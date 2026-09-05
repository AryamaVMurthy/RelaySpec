# Contributing

Start with the [research status](README.md#research-status) and
[current repair plan](docs/plans/2026-09-05-iclr-submission-repair-plan.md).
Install with `uv sync --locked`; run `make format` and `make check` before
committing. See [development commands](docs/DEVELOPMENT.md) for manuscript checks.

Keep algorithm changes separate from formatting or artifact migration. Add
focused tests for changed behavior. Do not edit archived source snapshots to
make them agree with newer code.

Every experiment should record its configuration, model revisions, source
revision or snapshot, dataset manifest, seed, and timing scope. Save raw paired
requests before producing summaries. State the reference decoder for speed
and output-agreement comparisons. Use a new run ID when rerunning; preserve
negative results that explain a conclusion or a change in direction.

Use ignored `outputs/` for new local runs. Promote reviewed evidence into
`reports/` deliberately. Keep model weights, credentials, and environments out
of commits. GitHub CI checks code; manuscript formatting and scientific review
are separate. Never update visual-review approval flags or a PDF hash without
inspecting the corresponding PDF.

Describe changes and validation in pull requests, including effects on
research claims when changing training, verification, or measurement.
