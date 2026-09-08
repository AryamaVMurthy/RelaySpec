# Native Joint Drafting Lab

A separate experiment investigating whether joint token-prediction training of a compact conditioning interface and native speculative drafter can improve on original DFlash. This project has its own code, configuration, results and decision record. It does not import RelaySpec, train a cross-model mapper, or modify the RelaySpec manuscript.

## Question and comparison

The target is frozen Qwen3-8B. The baseline is its pinned released native DFlash-b16 drafter. Students initialize from that native checkpoint, reduce the conditioning taps and/or draft depth, and train both the conditioning projection and draft transformer. This is native compression fine-tuning, not training from scratch or a claim of architectural novelty.

Four initial controls use identical data, updates, seed and evaluation:

1. Two target taps, five draft layers; update the interface only.
2. Two target taps, five draft layers; jointly update interface and drafter.
3. Two target taps, three draft layers; jointly update interface and drafter.
4. Original five taps and five draft layers; additional native training control.

Also decode the untouched original and the compact student before fitting. Keep the target, token embeddings and vocabulary head frozen. Use the exact pinned original greedy verifier. Compare accepted progress, complete generation wall time, output arrays and truncation, not only training loss.

## Training alignment

For a sequence of tokens `x`, choose an anchor at position `a`. Target context features contain only positions `[0,a)`. Draft inputs are `x[a]` followed by 15 masks. Draft output positions 1..15 predict tokens `x[a+1:a+16]`. Their target soft labels are obtained from target final hidden states at `[a,a+15)`, each of which predicts the next token. No future target feature reaches a draft input.

The initial objective is full-vocabulary KL(target || student), temperature 1, with position weights `exp(-j/7)` for `j=0..14`. Later comparisons include data cross-entropy, hard target labels, mixed losses, uniform weights and other decays. These objectives are prediction surrogates; true accepted progress is measured in decoding.

Training examples come from pinned NuminaMath records and solutions. Frozen target activations are independently generated for this experiment. Existing raw dataset storage may be read, but no previously trained cross-model mapper or feature target is used. Training and fitting-validation records are disjoint by normalized question. Development benchmark exposure is recorded; final confirmation requests must be frozen separately after selection.

## Research stages and completion requirements

1. Pipeline gates: causal alignment, frozen target/head gradients, both interface and draft updates, cache/decoding compatibility, reproducible controls and finite loss.
2. Four-arm joint-training pilot, then matched learning-rate and loss sweeps. Cover rates 2e-6, 1e-5, 5e-5 and 2e-4 where stable; record divergent settings.
3. Scale promising configurations across data (128, 512, 2,048 and 8,192 records), optimization (128, 512 and 2,048 updates), and architecture (2/3/5 taps and 3/4/5 draft layers). Use adaptive screening with recorded rejection reasons instead of silently omitting unfavorable runs. Extend an improving trajectory rather than treating a short failed fit as decisive.
4. Repeat selected comparisons across at least three training seeds; include the additional-training original-architecture control. Tune block lengths on development requests.
5. Freeze a final candidate and evaluate original DFlash and the candidate on at least 128 separate requests at a 2,048-token output cap, with repeated timing, output agreement and quality/truncation audit. Test math, code and dialogue generalization separately.
6. Report a bounded decision: faster with supported output quality, no supported gain within this search, or unresolved. A pilot alone cannot close the goal. Preserve all configs, source hashes, raw results, checkpoints, negative outcomes and a reproducible report with graphs.

Use at most four GPUs concurrently. Initial four-GPU jobs have independent lanes capped at 540 seconds. Keep at least two lanes as experiments completing within ten minutes, including during longer follow-up training.

## Sources

- Native implementation: https://github.com/z-lab/dflash, pinned commit `94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756`.
- Native training reference: https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dflash-speculative-decoding (official software documentation, accessed 2026-09-08). We implement the documented anchor/mask alignment in the pinned inference backbone; this is not a reproduction of its full training pipeline.

## Run

Local tests: `PYTHONPATH=. /home/aryamavmurthy/work/RelaySpec/.venv/bin/python -m pytest tests -q` from this folder.

Cluster launch: `sbatch --export=ALL,WAVE=configs/pilot.json run.sbatch` from an immutable uploaded source snapshot. The launcher creates separate per-lane configs and logs. Check every lane result, not only the Slurm exit status.
