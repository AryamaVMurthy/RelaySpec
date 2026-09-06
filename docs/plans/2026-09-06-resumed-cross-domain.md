# Resumed cross-domain experiments

User explicitly authorized both held-out confirmation and math-only versus mixed-domain training, removing the prior 90-minute total limit. Retain four GPUs maximum and under-ten-minute pilots. Large distinct-data scaling remains paused.

1. Finish exact-cache composition pilots: four parallel fits per arm, 16 updates, common 2048-record validation with separate math/general diagnostics, followed by duplicate-checkpoint decoding. Audit raw records before promotion.
2. Freeze existing dense N512 and N2048, seed1729, step8192, for DFlash8 and EAGLE8. Compare each family with its native AR, native drafter, and source-reuse controls. Pilot on eight previously exposed GSM8K questions; then evaluate the frozen256 primary questions at2048 output tokens. Do not use reserve or tune on confirmation. Record paired throughput, output lengths, cap counts and conservative paired accuracy intervals; keep the1pp quality margin even if inconclusive.
3. Fit composition arms at8192 updates, saving2048 and8192. Four fits per arm: dense, factorized512, MLP512, dense seed1730. Same2048 distinct records, objective and update count; math-only versus1024math+1024general. Report actual tokens because record matching is not token matching. The common validation cannot select horizons.
4. Compare all eight composition endpoints together on exposed math, code and conversation inputs, with matched AR/native/source references. First use a short operational pilot, then a declared fixed evaluation. Distinguish math accuracy from unscored code/conversation throughput; no unsupported quality claims.
5. Audit all raw results, regenerate paper assets, revise findings and limitations, rebuild and review the paper. Keep frozen-confirmation results separate from composition development comparisons.

The older timebox completion record remains a historical record, not an active scheduling limit. Current jobs are recorded under reports/mapper-scaling-20260905/resumed-composition and resumed-confirmation.
