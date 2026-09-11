# AUF mapper reproduction handoff

**Start here.** Reproduce the final, matched standalone **fusion mapper** results for GSM8K, KiCad and NanoCoder. Do not train the target LoRA or the drafter body. No MoE, router, SVD, perpendicular projection, or interpretability experiments are required.

## What is included

- `code/`: original AUF objective, training/export logic, feature capture, safe vLLM weight sharing and request timing; small path/runtime wrappers make these usable outside Turing.
- `data/`: exact 4,096 selected training prompts and 128 evaluation prompts per domain, including messages and stable IDs. `*_references.json` contains the corresponding target-AR output IDs and finish reasons.
- `expected_results.json`: measured native/full-mapper results and fitting times, for comparison—not a replacement for your own timing controls.
- `provenance/`: model/data revisions, environment versions, source origins, and package file checksums.

Model weights, trained mapper checkpoints, generated **training** responses and feature caches are not bundled. Download pinned models and regenerate training responses/features, or obtain the existing caches from the original project. Prompt manifests alone do not give bitwise-identical training trajectories across arbitrary software/hardware.

**Packaging validation:** syntax and manifests are checked in this workspace; the packaged loss has a CPU check recorded in provenance/packaging_validation.json; these newly portable wrappers have not been run through a fresh GPU training/evaluation cycle. The originating experiment code was used for the reported results. The recipient's Codex should run the small checks below before scaling, especially when changing vLLM versions.

## Models and data

All targets are **BF16 `Qwen/Qwen3-4B` plus the selected frozen target LoRA**. The unmodified draft is `z-lab/Qwen3-4B-DFlash-b16`. NanoCoder uses the adapter tokenizer; Math/KiCad use the base tokenizer. Thinking is disabled for every generated prompt.

Pinned repository IDs and revisions are in `provenance/models_and_data.json`. Use the bundled message selections rather than reconstructing splits: this preserves system prompts, formatting, ordering and exclusions. The publisher's assistant references are not training labels; **active-target-generated tokens are the labels**. Training/evaluation prompt IDs are disjoint. This does not certify that target adapters never saw related source data.

SpecForge: `https://github.com/sgl-project/SpecForge.git`, commit `953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef`. Use its DFlash blocks, attention mask, anchor selection and label alignment unchanged. AUF is our formula implementation atop this upstream code, **not author-released AUF code**. AUF is prior work; do not claim the loss itself as novel (Spec-AUF: https://arxiv.org/abs/2607.01893).

## Architecture and loss

Concatenate five target hidden states after zero-based transformer layers `[1,9,17,25,33]` (vLLM boundary indices `[2,10,18,26,34]`):

\[
h\in\mathbb R^{12800},\qquad z=\operatorname{RMSNorm}\left((F_0+BA)h\right).
\]

`F0` has shape `[2560,12800]`. `A` is `[56,12800]`; `B` is `[2560,56]`. PEFT rank=alpha=56, so alpha/rank=1. Train **860,160 parameters** in fusion `fc` only. Initialize A with PEFT's seeded initialization, B to zero. Freeze the draft body, native embedding/head, original fusion weight and RMSNorm (gain and epsilon). No extra trained bias. Forward arithmetic is BF16, trainable factors FP32. Export folds `F0 + BA` in FP32, then casts to BF16; inference does not need separate low-rank matmuls.

Each sampled block contains one clean response-token anchor and 15 masked continuation slots, mask token ID **151669**. Slot j predicts the actual target-generated token at that position. Use SpecForge's original context/block attention mask—do not substitute causal next-token prediction or ordinary SFT.

For valid continuation mask m and draft probabilities q, define

\[
a_j=\mathbf1[\arg\max_v q_j(v)=y_j],\quad
w_j=\operatorname{stopgrad}\prod_{i<j}(1-m_i+m_i a_i),\quad
\mathcal L_{\rm AUF}=\frac{\sum_{b,j}m_{bj}w_{bj}[-\log q_{bj}(y_{bj})]}{\sum_{b,j}m_{bj}w_{bj}}.
\]

The clean anchor, padding and out-of-range positions have m=0 and do not terminate the prefix. **The first wrong valid proposal is included; later proposals get zero weight.** Example over eligible proposals: correctness `[true,false,true]` gives weights `[1,1,0]`. First wrong gives `[1,0,...]`; all correct keeps every valid label. No gradient flows through argmax/support. No KL, hidden MSE, exponential decay, auxiliary selector or drafter-body adaptation.

Normalize accumulated chunk numerators/denominators within each microbatch. DDP and gradient accumulation then average the microbatch-normalized losses equally; do not silently change this to one globally token-normalized loss. `code/objectives.py` contains the actual implementation.

## Training recipe (same for all three domains)

| Setting | Value |
|---|---|
| Unique training prompts | 4,096 |
| Generated response cap | 4,096 tokens; natural EOS may be shorter |
| Feature storage | Dense BF16 states at every saved prompt/response position |
| Loss supervision | Up to 8 sampled distinct response anchors per presentation, not dense anchor tiling |
| Updates / presentations | 2,000 / 16,000; repeat the same 4,096 examples |
| GPUs / batch | 2 GPUs; microbatch 2/GPU; accumulation 2; effective batch 8 |
| Optimizer | AdamW, lr 1e-4, betas (0.9,0.999), epsilon 1e-8, weight decay 0 |
| Schedule | 100-step linear warmup, then cosine over the fixed 2,000-step horizon |
| Clipping / dropout | Global gradient norm 1; LoRA dropout 0 |
| Seed / checkpoint | 42; fixed final checkpoint, no dev-set selection |

Keep the source batching/RNG schedule to closely reproduce the fit: 128 shards of 32 examples, shuffled by `Random(42+epoch)`; within-shard shuffle followed by stable length-bin sorting (`length//64`); deterministic rank assignment. Before each forward, anchor RNG seed is `42*1000 + epoch*10000 + rank*3000 + microbatch_index`. See `train.py`. A features shard is `{'rows': [32 rollout records], 'features': [32 BF16 tensors]}`; tensor shape is `[len(full_ids),12800]`. Each row stores `group_id`, `prompt_token_ids`, `output_ids`, `full_ids`, `text`, `finish_reason`. Preserve target features and tokens from the **same** prefix.

## Minimal execution sequence

Use a fresh output directory per domain. Obtain the versions in `provenance/environment.json` using compatible CUDA wheels; these version strings are a recorded environment, not a promise that every package is on the default PyPI index. No Turing credentials or Slurm access is needed.

```bash
# Run from the handoff directory; substitute your local absolute paths.
export HANDOFF="$(pwd)"
export AUF_DOMAIN=math                    # math, kicad, nanocoder
export AUF_WORKDIR=/your/work/math
export AUF_MODELS=/your/models/qwen3_4b   # contains target/ and draft/ HF snapshots
export AUF_ADAPTER=/your/models/math_adapter
export SPECFORGE_ROOT=/your/src/SpecForge
export PYTHONPATH="$HANDOFF/code:$SPECFORGE_ROOT"
export VLLM_BATCH_INVARIANT=1
export VLLM_USE_V2_MODEL_RUNNER=0
export AUF_SAFE_RUNTIME=1
mkdir -p "$AUF_WORKDIR/setup"
cp "data/${AUF_DOMAIN}_train.json" "$AUF_WORKDIR/setup/train.json"
cp "data/${AUF_DOMAIN}_eval.json" "$AUF_WORKDIR/setup/eval.json"

# Generate once using the active target LoRA; then prefill those saved trajectories.
python code/generate.py --tag full --count 4096 --cap 4096
python code/capture.py --tag full
python code/check_inputs.py

# Two-GPU mapper fitting; never resume into an unrelated pre-existing workdir.
unset AUF_SAFE_RUNTIME
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  code/train.py --kind fusion_r56 --tag full --epochs 4 --max-steps 2000

# One GPU per standalone measurement. Run native and mapper on the same full cohort.
export AUF_SAFE_RUNTIME=1
CUDA_VISIBLE_DEVICES=0 python code/evaluate.py --variant native \
  --reference-file "data/${AUF_DOMAIN}_references.json"
CUDA_VISIBLE_DEVICES=0 python code/evaluate.py --variant mapper \
  --reference-file "data/${AUF_DOMAIN}_references.json"
```

**Before those full commands:** validate a handful of prompts (short response, long response, EOS, KiCad length stop), capture causality/alignment and a two-update fit on one 32-row shard. Check only the fusion factors change and folded export matches the trained layer to BF16 tolerance. Run `python code/check_loss.py` for the AUF edge cases and gradients. Manually inspect representative outputs; stop on any disagreement. `train.py --tag check --epochs 1 --max-steps 2` expects that shard under `features/check/`; use a separate work directory. Do not describe a changed prompt, verifier or loss implementation as upstream official.

`evaluate.py --count 2` is convenient for a smoke test but does not guarantee coverage of all ending/length types. Use representative selections in a separate gate workdir. The wrapper stops if any target-reference token/finish differs. If reproducing on different hardware/software, first generate fresh AR references with the same adapted target, audit disagreements against bundled references, and then use the new declared references consistently; do not silently relax equality or alter the verifier. The wrapper's `--variant ar` can produce AR outputs but still checks the supplied references.

## Timing and expected results

Standalone **one request at a time**, vLLM on NVIDIA L40S, 128 heldout prompts/domain, temperature=0, top_p=1, top_k disabled, seed=0, thinking off. Evaluation response caps: **Math 2,048 / KiCad 8,192 / NanoCoder 2,048**. Fifteen speculative proposals. Prefix caching off; async scheduling off; FLASH_ATTN; compilation mode0; request FULL CUDA graphs (effective FULL_DECODE_ONLY in the recorded backend). Runtime configuration is in `code/runtime.py`.

The provided safe-weight-sharing patch is essential: share frozen base embedding/head **weight storage** with the draft, never the target's LoRA wrapper/metadata. The selected adapters do not adapt embedding/head tensors. Verify this assumption if swapping adapters. Target verification and sampling remain vLLM's; AUF changes training only.

Time `engine.generate` per request after model load/warmup; collect counters outside the timer. Detect first-use JIT and retime that request after compilation. Preserve initialization, warmup and cold-request times separately. Speedup is `sum(native request seconds) / sum(mapper request seconds)` on identical prompt IDs and outputs—not the average of per-prompt ratios. Do not reuse our measured times as your own controls.

| Domain | Mapper/native speedup | Native / mapper acceptance length | Mapper fit time (2 L40S) |
|---|---:|---:|---:|
| Math | 1.250× | 4.477 / 5.691 | 3.83 min |
| KiCad | 1.337× | 5.953 / 7.912 | 10.51 min |
| NanoCoder | 1.053× | 5.042 / 5.356 | 4.54 min |

Acceptance length here is `1 + accepted_draft_tokens / verification_iterations`, including the verifier/bonus token. Also report draft-only length, accepted/proposed fraction, position survival, tokens/s, sum latency and exact-token agreement. These are speedups over **native speculative decoding**, not AR. Fit times exclude rollout generation, feature extraction and evaluation. Different GPU UUIDs are fine; control GPU type and benchmark settings. Small speed differences require repeats; paired-prompt bootstrap intervals alone do not cover hardware or seed variation.

## If reusing original caches

See `provenance/original_paths.json`. Those paths require access to the original Turing project and are not portable dependencies. Math/KiCad final weights are `seen_16000`, which means processed presentations, **not 16,000 unique examples**. NanoCoder here uses the sampled 4,096-example AUF checkpoint, not its older 18,016-example/16,384-cap dense experiment. No private credentials are included.
