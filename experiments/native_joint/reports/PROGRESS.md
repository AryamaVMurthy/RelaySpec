# Goal and progress ledger

Goal: investigate the standalone native joint-training experiment in depth, vary optimization/objective/data/architecture, and reach an evidence-based decision against original DFlash. The staged requirements are in the experiment README. The goal remains active; no pilot closes it.

## Completed

- Created a separate project in `experiments/native_joint`, without importing RelaySpec code or changing its manuscript.
- Pinned native Qwen3-8B target/drafter and original verification routine.
- Added native compact student construction, offline target feature capture from raw Numina solutions, correctly shifted masked-token KL training, paired before/after decoding and source/checkpoint provenance.
- Four local tests pass: tokenizer compatibility, no future conditioning features, teacher-detached/student-active gradients, and independent student construction.
- Job **29152**: engineering failure before training because this Transformers version returned a BatchEncoding from `apply_chat_template`. Preserved logs, fixed canonical token extraction, added regression test. No scientific result.
- Job **29153**: four-lane smoke passed, eight training and two validation records, four updates, two development requests at a 64-token cap. Joint arms update both interface and draft; interface-only leaves draft weights unchanged; target has no gradients. Peak allocated GPU memory 25–38 GB. These are pipeline checks.
- Job **29154**: four-arm pilot passed, 128 training and 16 validation records, 128 updates with accumulation 2 (256 sampled blocks), eight GSM8K development requests at a 512-token cap. Joint two-tap throughput 94.9% of original; three-layer/two-tap 47.3%; additional training of original architecture 101.2%. All student capped outputs matched native in 8/8 requests. Small point gains are not yet confirmed.

## Protocol correction and recovered evidence

Job 29155/lane2 failed a checkpoint reproduction gate after training. Whole-model `.bfloat16()` had rounded both nonpersistent RoPE frequency buffers; they are omitted from state_dict and are reconstructed in FP32 on reload. A focused test reproduced this, and parameter-only dtype conversion fixed it. Original post-training results from 29153/29154/29155 are excluded from promotion (see protocol-issue.json). Training weights remain valid.

Jobs **29157** and **29158** re-evaluated the eight pilot/rate checkpoints with original buffers and paired rounded-buffer diagnostic arms. Every corrected checkpoint passed reload and duplicate gates. Corrected results are in CORRECTED_RESULTS.md; the higher target-KL learning rates reduced validation loss but worsened decoding. The original-architecture fine-tune's 1.3% point gain has an interval spanning parity.

Job **29159** passed all four lanes with the corrected training pipeline: 512 records, 32 validation records, 512 updates, two examples per update. Joint two-tap five-layer throughput retained 94.3% with target KL, 97.4% with native-drafter KL, 93.6% with equal teacher blending, and 93.0% with higher-rate blending. All 8/8 outputs matched native per arm.

Job **29160** passed all four lanes: native-KL joint five-layer at rate5e-5 retained96.2%; four-layer drop3 and drop2 retained76.8% and76.9%, improving from66.7% and72.3% before training. Matched native-KL interface-only reached97.8%. All8/8 outputs matched native. These remain eight-request, single-seed development screens.

## Active

Job **29161** completed feature preparation: all four lanes passed in approximately three minutes. Merged `data/prepared-8192.json` verifies8,192 training +256 validation records, unique normalized questions and cross-split disjointness. Exact token IDs and cache hashes are preserved in run29161 artifacts. This is prepared data, not yet an8,192-record fit.

Job **29162**, project `/home/aryama.murthy/native-joint-20260908-v8` on Turing, is the active four-GPU wave:

- lane0: native-KL joint two-tap/five-layer,2,048 records,1,024 updates, rate1e-5.
- lane1: matched interface-only,2,048 records,1,024 updates, rate5e-5.
- lane2: native-KL four-layer/drop3,512 records,2,048 updates, rate5e-5.
- lane3: native-KL four-layer/drop2,512 records,2,048 updates, rate1e-5.

The first two lanes are capped at540seconds; the longer fits at900seconds. All use the prepared cache. The runner now validates hashes on first access, bounds activation RAM, tracks actual distinct records consumed, and requires all requested records to be used. `data-and-long-fit.json` is the exact wave. Check Slurm and lane statuses before taking further action; no restart based only on absent logs.

## Next required implementation and experiments

1. Collect and analyze job29162, including every lane result, validation trajectory, frozen controls and checkpoint reload gate. The8192-record cache is already collected/audited; cached tensors remain on `/scratch/aryama.murthy/native-joint/29161/laneN` (login access adds `/scratch/node07`).
2. Prepared-cache loading is implemented in v8. Reuse `data/prepared-8192.json` in subsequent configs. Investigate any failure before restarting; preserve learned checkpoints and partial curves.
3. Track actual distinct records consumed. Current accumulation2 ×2,048updates consumes at most4,096 records; do not call that an8,192-record fit. Add correctly masked variable-prefix batching or enough updates/accumulation to actually fit all requested records. Batched positional IDs must place masked queries at each example's true anchor; padding keys must be masked. Validate against separate single-block forwards.
4. Continue N2,048/N8,192 and longer-optimization studies with native-KL and original training controls, plus position-loss weighting and CE/mixed objectives as justified. Save validation-selected checkpoints for longer runs. Explore four-layer recovery before concluding it cannot work. Consider width/factorized-interface variations if they offer materially greater compute savings than dropping feature taps alone.
5. Multiple seeds, broader workloads, block-size tuning on development, and a frozen128-request/2,048-token final comparison remain required. No supported faster-native result yet; the goal remains active.

Offline token/prefix matches are diagnostics, not measured autoregressive speculative acceptance. Raw source solutions may be imperfect. Native-drafter KL conditions teacher and student on identical available prefix features and masks; target KL uses teacher-forced target next-token distributions. All new code and evidence stay in this separate experiment folder.
