# Reproducing experiments

## Configuration guide

- `configs/protocol_active/`: selected main fitting and evaluation settings.
- `configs/protocol_next/`: exploratory work, including experiments under review.
- `configs/design_selection/`: development ablations for configuration selection.
- `configs/breadth_ar/`: plain-target-decoding breadth configurations.
- Root configurations and numbered manifests: historical experiments.

Do not treat exploratory settings as validated defaults. Filenames, model
revisions, and snapshots preserve how recorded results were produced.

## Upstream drafter code

The runtime imports separate, pinned DFlash or DeepSpec checkouts. URLs and
checked revisions are in [the source log](research/relayspec-source-log.md)
and each configuration's `proposer.source_commit`. Set `DFLASH_SOURCE` or
`DEEPSPEC_SOURCE` to the corresponding checkout. The loaders verify its revision.

## Local paths

| Environment variable | Meaning |
|---|---|
| `RELAYSPEC_PROJECT_DIR` | Repository location used by cluster launchers |
| `RELAYSPEC_CACHE_DIR` | Storage for downloaded models and trained relays |
| `RELAYSPEC_OUTPUT` | One run's output directory |
| `RELAYSPEC_CHECKPOINT` | Override the saved relay path in benchmark configurations |
| `RELAYSPEC_INITIAL_CHECKPOINT` | Optional initial relay for supported training configurations |
| `DFLASH_SOURCE` | Pinned DFlash source checkout |
| `DEEPSPEC_SOURCE` | Pinned DeepSpec source checkout |
| `DEEPSPEC_PYTHON_OVERLAY` | Additional packages used by the historical EAGLE launcher |

Example setup, after configuring the upstream checkout:

```bash
export RELAYSPEC_PROJECT_DIR="$PWD"
export RELAYSPEC_CACHE_DIR="$PWD/outputs/cache"
export RELAYSPEC_OUTPUT="$PWD/outputs/my-training-run"
mkdir -p "$RELAYSPEC_CACHE_DIR" "$RELAYSPEC_OUTPUT"
export HF_HOME="$RELAYSPEC_CACHE_DIR/huggingface"
# Set DFLASH_SOURCE to your checkout at the configuration's pinned commit.
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_relay.py \
  --config configs/protocol_active/train_dflash_qwen3_8b_relative_4gpu.yaml
```

This is a GPU training command, not a CPU smoke test. Original runs use four
workers. For benchmarking, set `RELAYSPEC_CHECKPOINT` to the actual saved relay,
choose a new output directory, and use the appropriate benchmark script.

Slurm scripts retain original account, partition, module, and node settings.
Adapt those to your cluster or invoke the distributed Python launcher directly.
Historical absolute paths remain unchanged inside evidence snapshots.

## Recorded evidence

The [reports guide](../reports/README.md) identifies current review documents
and historical runs. Not every latest summary has its remote raw records
locally archived yet; see the submission review's artifact gap.

`make paper-assets` rebuilds the current pipeline from recorded summaries.
This does not repair the mixed-run comparisons identified in the review.
