# RelaySpec

**Retarget frozen speculative drafters with learned linear interfaces.**

RelaySpec converts internal features from a new target language model into the
input expected by an existing frozen drafter. The drafter guesses upcoming
tokens; the target checks those guesses. This can remove the need to run the
drafter's original source transformer during decoding.

The implementation includes DFlash and DeepSpec EAGLE-3 integrations. DFlash
uses a normalized conditioning vector; the selected EAGLE-3 relay preserves
feature scale. The two interfaces are deliberately different.

```text
Target features → learned linear interface → frozen drafter → target verification
       ↑                                                       │
       └──────────────── next decoding round ──────────────────┘
```

## Research status

This is an active research repository preparing an ICLR 2027 submission.
The manuscript and recorded results are **under scientific revision**.
The September 5 review identifies unresolved output-agreement questions,
objective/ablation mismatches, and a drift-checkpoint export defect. Passing
software tests does not establish that these research issues are resolved.

- [Current submission review and known issues](reports/ICLR_SUBMISSION_REVIEW_2026-09-05.md)
- [Detailed research execution plan and next experiments](docs/plans/2026-09-05-relayspec-evidence-execution-plan.md)
- [Time estimate and four-GPU Turing budget](docs/plans/2026-09-05-relayspec-time-estimate.md)
- [Paper framing, figure/table blueprint and result-presentation rules](docs/plans/2026-09-05-relayspec-paper-framing-plan.md)
- [Study of 42 public reviews across 11 related submissions](docs/research/2026-09-05-openreview-review-study.md)
- [Manuscript source](paper/iclr2027/relayspec_iclr2027.tex) and [draft PDF](paper/iclr2027/relayspec_iclr2027.pdf)
- [Evidence guide](reports/README.md)

The current manuscript reports the recorded source-reuse comparisons: **1.08–1.48×
MATH-500 throughput**, equal recorded task scores within each paired comparison,
and **7.63–7.80 GiB** lower peak memory in separate EAGLE-3 measurements.
Its asset pipeline checks all four main and sixteen breadth cells against saved
request records. The [rewrite and evidence-selection record](docs/plans/2026-09-05-relayspec-current-evidence-rewrite.md)
explains the current claim scope and which exploratory results require further
validation. These checks do not establish equivalence to plain target decoding.

## Install and test

Use Python 3.11 or newer. [uv](https://docs.astral.sh/uv/) manages the environment
and the checked-in dependency lock.

```bash
git clone https://github.com/AryamaVMurthy/RelaySpec.git
cd RelaySpec
uv sync --locked
make check
```

`make research-plan` validates the [research task ledger](configs/submission/README.md)
and shows the next work without launching experiments. `make submission-ready`
requires completion evidence for its required tasks; it currently reports that
research execution is incomplete. Neither command certifies scientific validity.

`make check` runs Python checks and CPU code/evidence tests without downloading
model weights. Full manuscript checks need LaTeX, Poppler, Ghostscript, and a
compiled paper. Their commands and known failures are in
[DEVELOPMENT.md](docs/DEVELOPMENT.md).

Optional dependencies are grouped by task:

```bash
uv sync --locked --extra paper  # plotting tools
uv sync --locked --extra data   # dataset preparation and external evaluation
uv sync --locked --extra drift  # experimental target-training tools
```

Do not interpret experiments using the current drift exporter until the
checkpoint issue in the submission review is addressed.

## Run experiments

Large-model runs require suitable GPUs, model access, pinned upstream drafter
checkouts, and configured cache/output locations. Start with the
[reproduction guide](docs/REPRODUCIBILITY.md). Existing cluster-specific paths
are preserved as provenance; they are not portable defaults for a new machine.

## Repository layout

| Directory | Contents |
|---|---|
| `src/relayspec/` | Relay modules, drafter integrations, decoding, metrics, evaluation |
| `scripts/` | Training, benchmarking, data preparation, analysis, and paper commands |
| `configs/` | Versioned data manifests and experiment configurations |
| `tests/` | CPU tests and separately marked manuscript checks |
| `paper/iclr2027/` | Current manuscript, bibliography, figures, and generated tables |
| `reports/` | Recorded experiment evidence, reviews, and historical summaries |
| `docs/` | Setup, development, provenance, and the current repair plan |
| `docs/archive/` | Superseded plans and the earlier research report |
| `slurm/` | Original launchers; adapt resource directives to your cluster |

See [CONTRIBUTING.md](CONTRIBUTING.md), [THIRD_PARTY.md](THIRD_PARTY.md), and
[the cleanup record](docs/CLEANUP.md).
