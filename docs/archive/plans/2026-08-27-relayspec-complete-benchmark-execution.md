# RelaySpec Complete Benchmark Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce the complete, reproducible evidence package needed to decide whether RelaySpec is an ICLR-quality inference paper: exactness, absolute speedup over native autoregressive decoding, controlled acceptance recovery, task quality, component timing, serving behavior, and minimal causal ablations.

**Architecture:** Keep Qwen3 targets and the DFlash proposer frozen. A separately trained relay maps hidden taps already computed by a new target into the conditioning interface expected by a proposer trained for another target. Every candidate is verified by the new target. Ordinary greedy speculative acceptance is algorithmically target-preserving, but block verification and one-token decoding use different floating-point kernel shapes and are not byte-identical in the current Transformers implementation. Therefore the primary quality comparison is task accuracy, with sequence equality reported as a diagnostic. The relay-specific quality gate is no measurable loss relative to source-reuse DFlash; the speed question is whether removed source-trunk time exceeds any loss in accepted tokens.

**Tech Stack:** Python 3.12, PyTorch, Transformers, official z-lab DFlash, CUDA events, NCCL/torchrun, pytest, Slurm on exactly four RTX 6000 Ada GPUs.

---

## 1. Fixed scientific contract

### Models

| ID | Frozen proposer | New target | Purpose |
|---|---|---|---|
| D-14 | `z-lab/Qwen3-4B-DFlash-b16` trained for `Qwen/Qwen3-4B` | `Qwen/Qwen3-14B` | Immediate scale-transfer result; current relay checkpoint exists |
| D-8 | same Qwen3-4B DFlash proposer | `Qwen/Qwen3-8B` | Decisive controlled transfer |
| Native-8 ceiling | `z-lab/Qwen3-8B-DFlash-b16` | `Qwen/Qwen3-8B` | Measures the acceptance/speed ceiling of target-specific proposer training |
| E-8/E-14 | Qwen3-4B EAGLE-3 proposer from DeepSpec | Qwen3-8B / Qwen3-14B | Second proposer-family generality, after DFlash results pass |

Each proposer/target pair receives its own relay. There is no universal-adapter claim.

### Decoding

- Qwen3 non-thinking mode only.
- Temperature `0.0`; greedy output is the main exactness claim.
- DFlash block size `16` in the main table.
- Maximum new tokens `2048` for the full task suite.
- Stop at EOS; record cap-hit rate and actual output length.
- Exactly four visible GPUs for every training or benchmark job.

### Main methods

1. `native_ar`: new target alone.
2. `native_target_dflash`: target-specific official proposer, available for D-8.
3. `naive_source_reuse`: old 4B source trunk + frozen 4B proposer + new target.
4. `relay_f`: feature-reconstruction relay.
5. `relay_p`: feature warm start followed by frozen-proposer-aligned refinement; intended main method.

The checkpoint used by the live D-14 MATH-500 run is a two-stage feature-loss
checkpoint. Its legacy raw method label is `relay_p`, but it must be reported as
`relay_f_refined` unless and until a proposer-aligned objective is actually run.

### Full benchmark suite

| Category | Benchmark | Prompts | Cap | Score |
|---|---|---:|---:|---|
| Math | MATH-500 | 500 | 2048 | official exact-answer accuracy |
| Math | GSM8K fixed subset | 128 | 2048 | exact-answer accuracy |
| Code | HumanEval | 164 | 2048 | EvalPlus-compatible pass@1 |
| Code | MBPP fixed subset | 200 | 2048 | EvalPlus-compatible pass@1 |
| Chat | MT-Bench | 80 | 2048 | token equality, latency, length; native quality scored once |

AIME is excluded. The first full run is MATH-500 because its manifest and trained D-14 relay already exist.

## 2. Minimal checks before expensive runs

Only these checks gate the full benchmark:

1. Unit tests pass.
2. Slurm allocation reports exactly four CUDA devices.
3. Eight fixed MATH-500 prompts run through `native_ar`, `naive_source_reuse`, and `relay_p` at 256 tokens.
4. Source-reuse DFlash and RelaySpec match each other on all eight prompts; equality to native AR is recorded but does not gate the run because block and single-token target kernels can differ numerically.
5. Timed regions explain at least 95% of speculative decode time and no NaN/zero-output record is produced.

Do not add unrelated random probes.

## 3. Exact run order

### Stage A — valid D-14 absolute benchmark

1. Extend the current harness from four prompts to arbitrary fixed manifests sharded over four ranks.
2. Add native AR to the paired method loop.
3. Store one JSON record per `(prompt, method, repetition)` with output hash, output length, task answer, correctness, TTFT, decode time, request time, target calls, acceptance, and component CUDA timings.
4. Run the eight-prompt smoke job.
5. Immediately submit complete MATH-500 with 500 prompts, one measured repetition, one warmup, and a 2048-token cap.
6. Score all three outputs with the pinned Qwen math evaluator, aggregate paired bootstrap 95% intervals for throughput ratios and accuracy deltas, and report all sequence-equality rates.

Headline Stage-A rows:

- Qwen3-14B native AR.
- Qwen3-4B DFlash source trunk + Qwen3-14B verifier.
- Qwen3-4B DFlash + RelaySpec + Qwen3-14B verifier.

### Stage B — controlled D-8 ceiling

1. Cache pinned Qwen3-8B target and official Qwen3-8B DFlash proposer on the allocated node scratch.
2. Generate/cache 4,096 mixed non-thinking training anchors: 50% math, 25% code, 25% chat.
3. Train D-8 `relay_f` from scratch.
4. Refine it as `relay_p` through the frozen proposer using prefix-survival/proposal-aligned loss.
5. Compare native AR, native 8B DFlash, naive 4B-to-8B reuse, relay-f, and relay-p on the same 128-prompt development manifest.
6. Freeze the best relay before viewing full benchmark results.
7. Run the complete five-benchmark suite.

Primary controlled result: relay-p must recover at least 90% of native 8B DFlash mean committed tokens/cycle while using less than 5% of its reported/reproduced training compute.

### Stage C — final D-14 result

1. Apply the D-8-selected relay architecture and objective to D-14; only target-specific dimensions/taps change.
2. Train on the same fixed 4K mixture; use 16K only if the prespecified data-scale ablation improves held-out tokens/s.
3. Run all five benchmarks for native AR, naive reuse, relay-f, and relay-p.
4. Run three final relay-p seeds on the 128-prompt development manifest and report mean/range.

### Stage D — minimal ablations

Run only on the frozen 128-prompt D-8 development manifest unless noted:

1. Objective: feature-only versus proposal-aligned refinement.
2. Relay: linear versus low-rank residual.
3. Target taps: one final tap versus three depth-spaced versus five depth-spaced.
4. Data: 1K versus 4K versus 16K.
5. Block size: 8 versus 16 versus 32, including full cycle cost.
6. Source removal: source weights loaded versus unloaded, to separate latency and memory benefits.
7. Scale: D-8 versus D-14.

Select by held-out end-to-end tokens/s, not feature MSE alone.

### Stage E — proposer-family generality

After D-8 and D-14 pass:

1. Implement the same frozen-interface abstraction for the DeepSpec Qwen3-4B EAGLE-3 proposer.
2. Train separate E-8 and E-14 relays.
3. Run the 128-prompt mixed development set and the full MATH-500 benchmark.
4. Report native AR, naive EAGLE source reuse, and relayed EAGLE under the same target engine.

### Stage F — serving and regime map

For the selected D-8 and D-14 methods, use 64 fixed prompts:

- prompt lengths: 128, 512, 2048, 8192;
- requested output lengths: 128, 512, 2048;
- concurrency: 1, 4, 8, 16 across the fixed four-GPU node.

Report TTFT, p50/p95 request latency, output tok/s, GPU memory, utilization, and the fraction of wall time removed by RelaySpec. This is the Amdahl-law regime figure.

## 4. Required metrics and statistical treatment

For every request:

- prompt/model/config identifiers and immutable revisions;
- input/output tokens, EOS/cap status, output hash and decoded answer;
- official task correctness;
- TTFT, decode time, end-to-end time, inter-token latency, output tok/s;
- committed tokens/cycle and survival by draft position;
- source-trunk, relay, draft, verifier, commit/crop, synchronization, and unattributed time;
- target/draft calls, peak allocated/reserved memory, GPU identity;
- exact equality to native AR and to source-reuse DFlash as diagnostics.

Aggregate with prompt-paired bootstrap 95% confidence intervals. Report geometric-mean speedup across benchmarks and arithmetic means only for directly additive counts/times. A paper speed claim requires the lower paired confidence bound above 1.0.

## 5. Paper tables and figures

1. Main D-8/D-14 benchmark: quality, exactness, acceptance, tok/s, native-AR speedup, naive-reuse speedup, memory.
2. Controlled D-8 ceiling: native DFlash versus relay-p, acceptance recovered, trainable parameters, examples, GPU-hours.
3. Component-time waterfall and measured-versus-predicted Amdahl speedup.
4. Minimal ablation table.
5. Prompt/output-length and concurrency heatmap.
6. Acceptance-survival curves.
7. EAGLE-3 generality table.

## 6. Concrete implementation tasks

### Task 1: Full paired benchmark harness

**Files:**
- Modify: `scripts/benchmark_relay.py`
- Modify: `src/relayspec/generation.py`
- Modify: `configs/benchmark_relay_qwen3_4gpu.yaml`
- Create: `configs/benchmark_relay_qwen3_14b_math500_smoke_4gpu.yaml`
- Create: `configs/benchmark_relay_qwen3_14b_math500_full_4gpu.yaml`
- Create: `tests/test_benchmark.py`

**Tests first:** sharding covers each prompt exactly once; method order rotates; aggregation uses native AR as equality reference; invalid/missing rows fail loudly.

**Verification:** `uv run pytest -q`.

### Task 2: Turing smoke and full D-14 submission

**Files:**
- Modify: `slurm/benchmark_relay.sbatch`
- Create: `slurm/benchmark_relay_full.sbatch`

**Verification:** eight-prompt job finishes with 24 records, RelaySpec agrees with source-reuse DFlash on 8/8 outputs, and all methods produce valid timing/output records; then full MATH-500 is submitted with a unique immutable output directory.

### Task 3: Scoring and aggregation

**Files:**
- Create: `src/relayspec/evaluation.py`
- Create: `src/relayspec/metrics.py`
- Create: `scripts/aggregate_results.py`
- Create: `tests/test_evaluation.py`
- Create: `tests/test_metrics.py`

**Verification:** official answer-extraction fixtures and synthetic paired-speed fixtures pass.

### Task 4: D-8 models, relay-f, and controlled ceiling

**Files:**
- Create: `configs/train_relay_qwen3_8b_4gpu.yaml`
- Create: `configs/benchmark_relay_qwen3_8b_controlled_4gpu.yaml`
- Modify: `scripts/train_relay.py`
- Modify: `scripts/benchmark_relay.py`

**Verification:** 128-prompt controlled report contains all five method rows and native equality for every speculative output.

### Task 5: Relay-p objective

**Files:**
- Create: `src/relayspec/losses.py`
- Create: `tests/test_losses.py`
- Modify: `scripts/train_relay.py`

**Tests first:** direct enumeration agrees with log-space prefix survival on blocks of length 2–4; only relay parameters receive gradients.

### Task 6: Full datasets and official scorers

**Files:**
- Create: `scripts/build_eval_manifests.py`
- Create: `configs/eval_manifest_full.json`
- Modify: `src/relayspec/evaluation.py`

**Verification:** manifest hashes, fixed subsets, decontamination log, and scorer versions are recorded.

### Task 7: Ablations, EAGLE-3, and serving

Implement only after the main DFlash gates pass. Every table cell must trace to raw JSONL, config hash, source revision, Slurm job ID, and prompt manifest.

## 7. Four-GPU schedule

| Phase | Four-GPU wall time | Outcome |
|---|---:|---|
| Harness tests + D-14 smoke | 0.5–1 h | Validity gate |
| Full D-14 MATH-500 | 3–8 h | First absolute paper result |
| D-8 caching + 4K relay-f/p training | 4–8 h | Controlled relay checkpoints |
| D-8 controlled 128 prompts | 1–2 h | Native-ceiling result |
| Full D-8/D-14 five-task suite | 12–24 h | Main benchmark table |
| Minimal ablations | 8–14 h | Causal evidence |
| EAGLE-3 + serving | 12–20 h | Generality and deployment |

Expected complete total: roughly 40–76 four-GPU wall hours, depending mainly on model downloads, output lengths, and whether EAGLE-3 integration is activated. The first decisive absolute-speed result is available after the D-14 MATH-500 job.

## 8. Immediate commands

Local checks:

```bash
uv run pytest -q
```

Sync only source/configs/tests to Turing:

```bash
rsync -avz --exclude '.venv' --exclude 'outputs' --exclude 'logs' --exclude 'reports' ./ turing:~/relayspec/
```

Submit smoke, then full run only after smoke passes:

```bash
ssh turing 'cd ~/relayspec && sbatch --export=ALL,CONFIG=configs/benchmark_relay_qwen3_14b_math500_smoke_4gpu.yaml slurm/benchmark_relay.sbatch'
ssh turing 'cd ~/relayspec && sbatch --export=ALL,CONFIG=configs/benchmark_relay_qwen3_14b_math500_full_4gpu.yaml slurm/benchmark_relay_full.sbatch'
```

The actual submission must also export the verified Python, DFlash source, cache, and relay checkpoint paths; no hidden fallback paths are allowed.
