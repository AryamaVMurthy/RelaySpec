# Competitor and Token-Length Scaling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run the most relevant feasible competitor comparisons and 4096/8192/16384/32768-token scaling using exactly four GPU workers per GPU allocation, at most four GPUs concurrently.

**Architecture:** Reuse the audited RelaySpec and pinned public PARD evaluators. Add a source-hash-guarded PARD-2 adapter and immutable benchmark configurations. Small allocation/correctness pilots precede full runs; completed results remain distinct from failed, censored, or unsupported cells.

**Tech Stack:** Python, PyTorch, Transformers, pinned public PARD, DFlash/DeepSpec, Slurm on Turing node07.

## Protocol

- Existing Turing jobs: none at start. Account verified as priyesh.shukla. node07 has four available GPUs.
- Competitors: first PARD-2 target-dependent and target-independent modes, original PARD and matched AR, plus RelaySpec/native/source controls on identical request manifests. Use released checkpoints; no need to retrain PARD-2. Keep runtime versions/attention/precision explicit. Never merge published speeds with measured speeds.
- EDA: assess compatible released architecture and weights before launching adaptation. Cross-size transfer is not equivalent to its same-size domain transfer; do not mislabel a substitute as EDA.
- TriSpec: adapter prior art must be discussed; full proxy verification changes the accuracy/latency contract. No unverified imitation counted as a competitor reproduction.
- Token length: user clarification requested for input context versus generated output length. Proceed with independent competitor implementation while waiting. Default if unanswered: input context sweep at exactly 4096/8192/16384/32768 tokenizer tokens and fixed 256 generated-token cap, reporting actual generated lengths, prefill, decode and peak memory. This is sequence scaling, not 32768 training examples. No new mapper fit is required.
- Length tests use deterministic, recorded natural-text contexts with a held-fixed task suffix. Treat them as controlled sequence-cost/retention measurements, not independent long-context quality benchmarks. Do not pad with special tokens or silently truncate. At 32K verify model position limit and include output/cache headroom.
- All pilots use four GPU workers and a ten-minute wall limit. Full jobs may be longer after a successful pilot, since the user removed the old 90-minute limit. Record job ids, configuration/source hashes, runtime, device, allocation, request outputs and acceptance traces.

## Task 1: PARD-2 evaluator

Files: modify `src/relayspec/pard_adapter.py`; create `scripts/benchmark_pard2.py`, pinned config and four-GPU Slurm script; extend adapter tests.

1. Add source-hash guard for the already pinned upstream PARD-2 file without changing proposal or verification logic.
2. Parameterize acceptance-trace checking by draft length, retain the old default for historical audits.
3. Test AST instrumentation and malformed traces locally.
4. Download pinned PARD-2 artifacts on node07 scratch via CPU setup job, verify weight hashes.
5. Run four-request pilot across TD/TI and matched AR, including unobserved timing and an identical verifier-observed replica.
6. Promote only after artifact and verifier checks pass to 128 development requests at 2048 output cap.

## Task 2: Sequence sweep

Files: create manifest/config builder in `scripts/`, configs under `configs/submission/competitors-length-20260906/`; minimally extend existing benchmark input handling if exact token manifests require it.

1. Declare the token axis and construct exact tokenized inputs with hashed provenance.
2. Test exact lengths, suffix preservation and context-window bounds.
3. Run one four-GPU pilot including the largest length, recording OOM/position-limit failures explicitly.
4. Run all four declared lengths using fixed checkpoints with AR/native/source/relay references. Independent requests are sharded across four workers; these are not tensor-parallel serving claims.
5. Audit paired rows and export separate prefill/decode/end-to-end/memory scaling summaries and graphs.

## Task 3: Results and remaining competitors

1. Collect terminal jobs and preserve all failures in the ledger.
2. Score math competitor outputs with the existing pinned scorer and report paired uncertainty.
3. Determine EDA feasibility from the actual code and document required model/architecture changes before training.
4. Update the paper only with completed audited comparisons, and separately correct TriSpec/PARD-2/EDA positioning.

## Execution update

- PARD-2 setup28190/pilot28196/full28201 completed; 384 paired rows and verifier replicas passed audit. Pinned math scoring completed.
- Context pilots28232/28233 failed at32K with CUDA OOM while baseline models co-resided. Their eight dependent sweeps were canceled automatically without GPU allocation.
- Revision b9dd3e6 isolates each method in a fresh four-worker process and unloads the unneeded DFlash source trunk for RelaySpec. Replacement pilots28256/28257 use15-minute safety limits, then full16-request cells have30-minute limits. No retraining or hidden truncation.
- Method order is sequential across processes, so isolated memory is interpretable but timing does not have within-request method-order rotation. DFlash AR still loads the small source drafter; report this memory overhead explicitly.
