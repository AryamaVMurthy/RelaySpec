# Experiment configurations

| Location | Role |
|---|---|
| `protocol_active/` | Selected main fitting and evaluation settings |
| `protocol_next/` | Exploratory settings, including experiments under review |
| `design_selection/` | Development ablations |
| `breadth_ar/` | Breadth experiments with a plain-target arm |
| `vocab_bridges/` | Tokenizer conversion metadata |
| Root YAML files and numbered manifests | Historical configurations retained for tests and provenance |

`relayspec_protocol.yaml` records earlier decisions and required artifacts;
it does not supersede the current submission review. Do not silently overwrite
manifests or rename historical experiment IDs. Use a new configuration for a
new experiment. Local path overrides are in [the reproduction guide](../docs/REPRODUCIBILITY.md).
