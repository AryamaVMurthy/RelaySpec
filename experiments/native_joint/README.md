# Native Joint Drafting Lab

A separate experiment investigating whether joint token-prediction training of a compact conditioning interface and native speculative drafter can improve on original DFlash. This project has its own code, configuration, results and decision record. It does not import RelaySpec, train a cross-model mapper, or modify the RelaySpec manuscript.

## Current speed target and rapid research mode

The user requires at least10% end-to-end speedup over original DFlash and has prioritized rapid, radical decoding ideas after joint training reached only near parity. `radical_decode.py` and `run_radical.py` test inference policies with the unchanged native weights: verification length, history lookup, conditional multi-branch verification, and recycling unused proposals. These screens extend the separate native study; they are not RelaySpec results. Every proposed token is still target-verified, and numerical output differences are measured explicitly. A small-screen gain is only a promotion signal: repeated broader tests and the reserved128-request/2,048-token comparison are required.

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

The native-teacher arm distills the frozen original drafter on identical prefix features and masked query tokens. Batched fitting right-pads conditioning keys, masks all padding, and preserves each query's true position. GPU gates compare padded and separate forwards in FP32 and BF16 and require padding perturbations to have no effect. Full-vocabulary losses average normalized per-block losses, so increasing batch size does not multiply the objective. Logs report distinct records actually consumed and supervised blocks. Optional validation-best checkpoints are saved separately; the standard after-training screen always evaluates the final checkpoint and labels it explicitly.

`data/screening.json` selects 16 requests per workload for adaptive work. `data/confirmation.json` reserves 32 other requests per workload (128 total), without overlap with screening IDs. These originated in an earlier project manifest; only separation from this standalone study's tuning is claimed. `evaluate_native.py` supports paired repeated timing, block-size comparisons, and optional original block=1 AR decoding. Confirmation requires an explicit frozen checkpoint/configuration and manifest match.

## Research stages and completion requirements

1. Pipeline gates: causal alignment, frozen target/head gradients, both interface and draft updates, cache/decoding compatibility, reproducible controls and finite loss.
2. Four-arm joint-training pilot, then matched learning-rate and loss sweeps. Cover rates 2e-6, 1e-5, 5e-5 and 2e-4 where stable; record divergent settings.
3. Scale promising configurations across data (128, 512, 2,048 and 8,192 records), optimization (128, 512 and 2,048 updates), and architecture (2/3/5 taps and 3/4/5 draft layers). Use adaptive screening with recorded rejection reasons instead of silently omitting unfavorable runs. Extend an improving trajectory rather than treating a short failed fit as decisive.
4. Repeat selected comparisons across at least three training seeds; include the additional-training original-architecture control. Tune block lengths on development requests.
5. Freeze a final candidate and evaluate original DFlash and the candidate on at least 128 separate requests at a 2,048-token output cap, with repeated timing, output agreement and quality/truncation audit. Test math, code and dialogue generalization separately.
6. Report a bounded decision: faster with supported output quality, no supported gain within this search, or unresolved. A pilot alone cannot close the goal. Preserve all configs, source hashes, raw results, checkpoints, negative outcomes and a reproducible report with graphs.

Use at most four GPUs concurrently. Initial four-GPU jobs have independent lanes capped at 540 seconds. Keep at least two lanes as experiments completing within ten minutes, including during longer follow-up training.

## Additional rapid directions

- `collect_progress.py` records native rollout draft features and target labels through the first rejection. `train_progress_head.py` fits small residual heads with CE, a margin plus native-preservation loss, or a differentiable prefix-progress surrogate. Post-rejection and post-EOS positions are masked. `analyze_progress.py` reports paired TPS, repair/break rates and protocol differences for every completed head fit.
- Prefix-conditioned refiners jointly train a second drafter pass on the tail after a known prefix. Copied prefix tokens are excluded from the loss. At inference only predicted prefixes are available; all proposals still go through target verification.
- Warm starts blend unverified previous draft or target-prediction tails into masked inputs. Target-tail recycling proposes previously computed target predictions directly, with fresh verification. These hypotheses do not treat rejected-prefix predictions as correct labels.
- Lazy target-head verification computes full-vocabulary logits in chunks until the first rejection is known, preserving its corrective token and the fully verified transformer cache. This changes matrix shapes and synchronization; numerical agreement and full latency are measured, not assumed.
- `profile_native.py` attributes instrumented native time with CUDA events; `analyze_profile.py` subtracts nested vocabulary-head spans to avoid double-counting. `collect_progress.py --config ...` can additionally save causal target features on native-generated sequences; `merge_rollout_sequences.py` merges only completed, disjoint caches for full-drafter fitting.
- `quantized_draft.py` packs an independent drafter and optionally a separate draft vocabulary head into int4. Its proxy routes target verification through the original BF16 target/head. Copy, frozen-control and checkpoint-reload gates precede timing. This tests standard quantization as a proposal-computation tradeoff, not an architectural novelty claim.

Run `analyze_radical.py` and `analyze_progress.py` from this folder to regenerate all positive and negative screens. `reports/PROGRESS.md` records the current queue and decisions. None of the completed adaptive screens yet establishes the required10% gain.

## Sources

- Native implementation: https://github.com/z-lab/dflash, pinned commit `94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756`.
- Native training reference: https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dflash-speculative-decoding (official software documentation, accessed 2026-09-08). We implement the documented anchor/mask alignment in the pinned inference backbone; this is not a reproduction of its full training pipeline.
- Quantization implementation: [TorchAO0.15.0 quantization API](https://github.com/pytorch/ao/blob/v0.15.0/torchao/quantization/quant_api.py), pinned to the [official Torch2.9.1 compatibility entry](https://github.com/pytorch/ao/issues/2919). The private dependency directory is separate from the shared training environment.

## Run

Local tests: `PYTHONPATH=. /home/aryamavmurthy/work/RelaySpec/.venv/bin/python -m pytest tests -q` from this folder.

Cluster launch: `sbatch --export=ALL,WAVE=configs/pilot.json run.sbatch` from an immutable uploaded source snapshot. The launcher creates separate per-lane configs and logs. Check every lane result, not only the Slurm exit status.
