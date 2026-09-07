# Llama speed optimization pass

Reuse the best established normalized-linear mapper; do not retrain in this pass. Keep the target and plain-AR baseline FP32 with TF32 disabled. Run the frozen drafter and source embedding/head in BF16, casting the FP32 mapper output to BF16 before the frozen drafter normalization. Target acceptance/commitment code is unchanged.

Four concurrent single-GPU lanes on node07 L40S:
- Llama-3.1-8B DFlash -> Llama-3.2-3B, block 10.
- Same transfer, block 16.
- Qwen3-4B DFlash -> Llama-3.1-8B, block 16.
- Same cross-family transfer, block 8.

All use the same eight confirmation prompts as job 28885, cap 1024, matched native AR in each lane. Compare to that full-FP32 run only after checking exact prompt sets, caps, and output hashes. Reject any setting with an AR mismatch. Job 28922, eight-minute wall limit, four GPUs total.

Training provenance: established maps configured for 4096 records, 1024 four-GPU updates, one example/rank/update, hence one pass through 4096 records. ZIP alternatives use 512 records, separately 3 epochs/54 updates or 20 epochs/360 updates. Training recipes and data budgets differ; these are practical available-checkpoint comparisons, not isolated loss ablations.
