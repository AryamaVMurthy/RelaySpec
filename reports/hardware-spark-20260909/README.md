# Completed GB10 hardware replication

MSI EdgeXpert MS-C931, NVIDIA GB10, aarch64. The original Qwen3-4B DFlash to Qwen3-8B RelaySpec mapper was frozen without fitting or tuning. This is a hardware replication on 32 previously evaluated requests, eight each from MATH, GSM8K, HumanEval and MT-Bench first turns. Batch one, BF16/SDPA, greedy, maximum 2,048 output tokens, two rotated timing repeats. No request reached the cap.

| Method | Tokens/s | Relative to AR | Device J/token | Isolated peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| AR | 11.14 | 1.00 | 3.767 | — |
| Native DFlash | 70.88 | 6.37 | 0.607 | — |
| Source reuse | 50.73 | 4.56 | 0.836 | 23.82 |
| RelaySpec | 60.92 | 5.47 | 0.705 | 17.42 |

RelaySpec is 20.1% faster than source reuse, with paired 95% request interval 17.1–23.2%, 15.7% lower sampled device energy per token and 6.40 GiB (26.9%) lower isolated peak allocation. Native DFlash remains faster overall. The code-workload point gain is 6.2%, with interval −0.6–11.8%, so a gain there is unresolved. No historical L40S timing is mixed into these estimates.

## Completed checks and profiling

- `main`: all 256 timed generations complete. Every second-pass token sequence and acceptance trace matches the first pass. RelaySpec, source reuse and native DFlash agree on all 32 outputs. Each agrees with AR on 7/32.
- Quality: each method scores 8/8 MATH, 7/8 GSM8K and 7/8 base HumanEval problems. No non-dialogue output remains unscored. The scoring input hash matches the completed first pass. MT-Bench has no judge score. Equal scores on this small sample do not establish quality equivalence.
- `memory-source`, `memory-relay`: fresh processes, four matched requests each, cap 2,048. Source retains only its 34 executed blocks. Relay retains inherited embeddings/head but removes the source transformer. All isolated outputs match the corresponding main-run outputs.
- `timeline-run`: four methods, one warmed MATH request, cap 256. Profiler annotations preserve tokens and acceptance traces. Source reconstruction accounts for 2.104 s GPU-projected time versus 31.789 ms for relay mapping. Relay target verification accounts for 4.443 s and drafting for 0.528 s. These spans are not unprofiled wall-time fractions or a union of kernel-busy time.
- `counters-source-run`, `counters-relay-run`: cap 64, eight filtered kernels per conditioning path. Source includes seven matrix products and one attention kernel. Relay includes eight matrix products. Raw CSV includes per-kernel time, achieved occupancy, compute and L2 counters. DRAM bandwidth counters are unavailable.
- `disagreement-01`: one identical-prefix native/AR mismatch reproduced at generated token index 15 (token 16). Native BF16 head outputs tie for “Understand”/“Determine”; FP32 head reprojection selects “Understand” for both captured BF16 hidden states. This is not full-FP32 decoding or a general exactness guarantee.

## Reproduction and provenance

The GPU harness is `experiments/hardware_spark` in the primary RelaySpec checkout. Every stage stores its actual executed wrapper and protocol. The main wrapper hash matches its provenance even though later isolated-memory and profiling stages use additional memory pruning and labels. Frozen decoder source hashes and model/checkpoint revisions are recorded in each protocol. No decoder weights were trained in this study.

Run `make paper-hardware-assets` to rebuild the hardware assets from this directory. In the shared worktree, pass `PYTHON=/home/aryamavmurthy/work/RelaySpec/.venv/bin/python`. `scripts/build_hardware_spark_assets.py` checks completion, request/repeat counts, quality input identity and profiler records. Its generated registry fingerprints plotted and tabulated inputs. The manuscript audit independently rebuilds and byte-compares these assets.

`profile-artifacts.json` records hashes and locations of the original Nsight binary reports and SQLite trace. These larger files remain in the primary checkout's `experiments/hardware_spark/runs/gb10-20260909/results` and on the remote machine. The CSV exports needed to reproduce the paper are retained here. `prior-visual-review.json`, `visual-review.json` and `render-comparison.json` document final PDF review and exact pixel identity for unchanged pages.

Device energy integrates one-second power samples with full interval coverage. It is not wall-socket energy. PyTorch allocated memory is not total system UMA occupancy. The timing run keeps all four methods resident, so only the separate memory runs support deployment-memory claims. Counter replay and timeline times are never used as headline throughput measurements.
