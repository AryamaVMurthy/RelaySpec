# Reproduce the single Qwen3-8B / mapped Qwen3-4B drafter experiment

This directory contains the code, pinned download recipe, reference outputs and provenance for **one configuration**: a frozen Qwen3-4B DFlash drafter transferred to Qwen3-8B using five added linear context maps trained on 16,384 examples. The two controls are Qwen3-8B autoregressive and Qwen3-8B with its native DFlash drafter.

There are **no model weights, trained checkpoints, tokenizers, feature tensors, wheel files or dataset shards in this distribution**. Every model and the source dataset must be downloaded at its pinned revision, and the mapper must be trained. The JSONL reference files contain prompts, generated token IDs and timings, not model parameters. Tensor checksums cannot reconstruct weights.

## Recorded result to confirm

| Pipeline | Output tokens | Sum of request seconds | Tokens/s | Speedup versus AR |
|---|---:|---:|---:|---:|
| Qwen3-8B AR | 124,899 | 4,760.891751 | 26.234371 | 1.000000× |
| Qwen3-8B + native 8B DFlash | 124,899 | 765.182257 | 163.227778 | 6.221906× |
| Qwen3-8B + mapped 4B DFlash | 124,899 | 745.356825 | 167.569405 | 6.387399× |

All three emitted **identical token arrays on all 128 prompts**, with the 2,048-token generation cap. There were 104 normal stops and 24 length stops. Mapper/native request-wall ratio: **1.026599×**. These are single-run measurements. GPU scheduling, clocks and runtime differences affect timing; exact matching timings are not a reproducibility criterion. `reference/results.json` includes paired-prompt intervals, which do not measure repeated-run or hardware variance.

## Hardware and software

Recorded host: Linux kernel 5.15.0-187-generic, glibc 2.35, x86_64. GPU: NVIDIA L40S, 46,068 MiB reported memory, driver **610.57.04**. Four GPUs were used to parallelize independent requests, not tensor-parallel model execution. See `provenance/hardware.txt` and `provenance/environment.json`.

| Component | Exact installed version |
|---|---|
| Python | 3.12.13 |
| vLLM | 0.28.0+cu129 |
| PyTorch | 2.13.0+cu129 |
| Transformers | 5.16.1 |
| Triton | 3.7.1 |
| safetensors | 0.8.0 |
| NumPy | 2.3.5 |
| PyArrow | 25.0.1 |

`requirements.lock` pins the installed dependency closure, not just these top-level packages. The core runtime versions were recorded during the experiment; the full dependency closure and hardware snapshot were read from the same environment when preparing this package. Package versions alone cannot fix all driver/compiler/queue-dependent floating-point behavior.

The CUDA 12.9 vLLM wheel is pinned by URL and SHA-256 in `provenance/vllm_wheel.json`. It is an upstream release artifact, not a bundled wheel. Plain `pip install vllm==0.28.0` can select a different CUDA build. Sources: [vLLM release](https://github.com/vllm-project/vllm/releases/tag/v0.28.0), [PyTorch CUDA 12.9 wheels](https://download.pytorch.org/whl/cu129/torch/).

Budget at least **500 GB free disk** for downloads and generated training features. The recorded train feature payload alone is **360,506,065,920 bytes** (336 GiB), before serialization overhead. Use at least 64 GB host RAM for each active capture/training process; generation and benchmarks used 8 CPU threads per GPU process. No original cluster access, account, credentials or absolute cluster paths are required.

## Install in a fresh checkout of this directory

Use a Python 3.12.13 interpreter. Installation and all generated files stay beneath this directory by default:

```bash
python3 verify_bundle.py
python3.12 bootstrap.py
.venv/bin/python -B -m unittest discover -s tests -p 'test_*.py'
```

`bootstrap.py` creates `.venv`, installs the exact CUDA wheel and lock, runs `pip check`, and verifies package versions. It does not download models. `TRANSFER_WORK=/absolute/path/to/scratch` can relocate subsequently generated work files. Set that environment variable consistently for every command; the default is `./work`.

## Rebuild end to end

Run commands from this directory. The environment setup in `reproduce.py` is part of the protocol. Do not globally enable batch invariance for training-data generation: its recorded engine used different settings from the final evaluation.

```bash
.venv/bin/python reproduce.py download
.venv/bin/python reproduce.py prepare
```

`download` acquires all four pinned model repositories (base 4B and its drafter; base 8B and its drafter) and the first pinned NuminaMath-CoT parquet shard. `pins.json` gives all repository IDs and full commit revisions. `prepare` reconstructs the original deterministic sample and asserts its hashes and exact evaluation token IDs against the references. It does not load reference token IDs as substitute model outputs.

### Generate 4,096-token training continuations

```bash
.venv/bin/python reproduce.py generation-check --gpus 0
```

Inspect `work/validation/probe_rollouts_r1/train/00000.jsonl`. Decode the token IDs with the freshly downloaded 8B tokenizer and inspect all eight samples against their prompts. Check ordinary EOS versus length stops, positive output length, no thinking content, and sensible prompt/continuation boundaries. This is a manual gate, not an automatic claim of correctness. If a discrepancy is unexplained, stop and investigate.

```bash
.venv/bin/python reproduce.py approve-generation
.venv/bin/python reproduce.py generate --gpus 0,1
```

The full generation uses **two workers**, 128 active sequences per engine, 1,024 queued prompts per call, and alternating 128-example shards. The 16 recorded queue windows are recoverable from `provenance/rollout_shards.json`. Shards are atomic and resumable; never change worker count or batch scheduling casually. Do not reuse the eight-check outputs as full-generation shards: their queue shape is intentionally smaller.

`check_rollouts.py` compares every regenerated shard hash with the historical shard hash. Training generation used normal BF16/compiled vLLM execution, **not the final batch-invariant evaluation configuration**. Its request scheduling can affect floating-point results. Therefore bitwise historical rollout recreation is **checked, not promised**. On mismatch, preserve the data and inspect versions, pins, prompts and queue settings. If deliberately continuing with a newly generated training realization, run:

```bash
.venv/bin/python src/check_rollouts.py --accept-new-data
```

This records the difference explicitly. It does **not** mean the historical training data/checkpoint were reproduced, even if the final target outputs later agree. There is no hidden pretrained mapper fallback.

### Capture and persist the 25% paired hidden states

```bash
.venv/bin/python reproduce.py capture-check --gpus 0,1
```

This constructs the index from regenerated rollouts, selects representative shortest/longest examples, runs sampled and dense capture on the bounded sample, and compares sampled rows with their dense counterparts for both targets. Inspect `work/cache/validation/comparison.json`, the sample group IDs and token-position boundaries before approving.

```bash
.venv/bin/python reproduce.py approve-capture
.venv/bin/python reproduce.py capture --gpus 0,1
```

Each target is teacher-forced on the **same complete 8B-generated token sequence**. Both persist the same sampled positions. Capture is native vLLM pooling with a small registered model subclass; it is not a Transformers decode loop. Features are written in 128-example CPU shards, with groups, positions, boundaries, rollout hashes and layer IDs. `check_features.py` verifies pairing, shapes, finiteness and full coverage before training.

### Fit and export the one mapper

```bash
.venv/bin/python reproduce.py train --gpus 0
.venv/bin/python src/check_mapper.py
.venv/bin/python reproduce.py export --gpus 0
```

Training uses 16,384 examples, three epochs, the final epoch checkpoint, and no development-set selection. `provenance/fit.json` records the observed losses, steps and parameter count. `check_mapper.py` compares tensor hashes with the historical mapper. A different training realization or numerical environment can cause this check to fail; preserve that distinction instead of claiming bitwise reconstruction. The exporter separately verifies the folded map numerically and rejects an error above the declared threshold.

The development partition is recreated only because it is part of the original deterministic split and supplies the original four warmup prompts. It is not used for optimization or checkpoint selection. Unused development features/rollouts are not generated by this package.

### Validate and benchmark all three pipelines

```bash
.venv/bin/python reproduce.py eval-check --gpus 0
```

Inspect the eight-prompt raw outputs, reference checks (`4` and `hello`), and `work/results/check.json`. All three arrays must agree, including length. No parser, normalization, token replacement or output forcing is used.

```bash
.venv/bin/python reproduce.py approve-eval
.venv/bin/python reproduce.py evaluate --gpus 0,1,2,3
.venv/bin/python reproduce.py verify
```

AR uses four independent 32-prompt workers. Native and mapped decoding use two independent 64-prompt workers each, concurrently across four GPUs. Each GPU processes one measured request at a time. The four warmup prompts are excluded. Requests affected by newly compiled kernels are rerun once, their cold time is retained, and repeated compilation fails the timing check.

Final verification checks all 128 original prompt token arrays, unique group coverage, equality among all three newly generated outputs, and equality to **the historical reference token arrays**. It reports request-wall and token-throughput speedups using freshly measured AR. It does not compare parallel job duration with summed request time.

## Source layout and portability

- `src/prepare.py`, `generate.py`: deterministic sample and target continuations.
- `src/capture.py`, `capture_runtime.py`, `sampling.py`: paired quarter-density context capture.
- `src/mapper.py`, `data.py`, `train.py`: the selected objective and optimizer.
- `src/export.py`, `mapper_runtime.py`: folded fusion and protected 4B embedding/head modules.
- `src/benchmark.py`, `metrics.py`, `verify.py`: native vLLM inference, timing and exact token audit.
- `IMPLEMENTATION.md`: formulas, tensor layouts, layer alignment and runtime contract.
- `provenance/source_map.json`: hashes of executed source files, extracted paths and declared packaging changes.
- `reference/`: only this final experiment's expected prompts, outputs and metrics.

The executable package is a **portable extraction of the selected computation paths**, not a byte-for-byte copy of whole repository scripts with unrelated branches. Cluster-specific paths/scheduler code were replaced, the single configuration was fixed, and export tensors are saved in one file instead of hard-linked shards. Training math, feature selection and final inference settings are retained. The exact provenance and numerical equivalence checks are documented rather than calling these packaging changes “official.”

## What was verified when packaging

The original experiment completed all three 128-prompt runs. The packaging work checks source syntax, reference equality, sampling boundaries, worker partitioning and the selected objective/folding on CPU. It **does not claim that this newly packaged download-to-training pipeline has itself been rerun on GPUs**. Read `provenance/package_validation.json` for the precise checks performed. `MANIFEST.sha256` covers every shipped file other than itself.
