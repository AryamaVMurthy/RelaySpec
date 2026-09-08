# Contribution clarity review

The main contribution list currently appears at the end of Introduction, split across rendered pages 1–2. Its first heading is concrete, but “Measured calibration regimes” and “Interface findings that inform deployment choices” name experimental categories rather than the conclusions a reviewer should remember. Replace these with explicit, evidence-linked claims. Keep a short synthesis in Discussion so readers see why the findings matter together.

## Suggested four-point contribution statement (about 190 words)

1. **Retarget the interface; preserve the drafter.** RelaySpec reuses released DFlash and EAGLE-3 prediction networks by learning their conditioning input from a new target’s features. This removes the source transformer from generation while retaining target-controlled verification, achieving 2.35–5.11× AR throughput in the primary Qwen suite (Section 4; Table 1).
2. **Useful transfer needs little target-specific calibration.** Frozen 512-example maps retain 97.5% and 98.9% of the corresponding 2,048-example throughput on 256 GSM8K questions. The controlled 16–32,768-record sweep identifies diminishing data returns in the tested setting (Section 5.7.1; Figure 2; Table 3).
3. **More expressive fitting need not produce better proposals.** Dense linear maps lead the completed long-output MLP comparisons, and feature regression leads the matched warm-budget CE/LoRA study. Lower feature error therefore cannot substitute for measuring decoding utility (Section 5.7; Table 4).
4. **The reuse boundary extends beyond within-family scaling.** Richer linear calibration reaches native DFlash throughput, with a repeated 2.6% advantage. Separate Llama and Qwen-to-Llama transfers reach 2.52× and 2.75× matched FP32 AR throughput and exact token agreement on 128 requests each (Sections 5.8–5.9; Tables 6–7).

## Interpretation and scope

- Point 1 is the method/deployment contribution. Adapter-only fitting, linear stitching, and generic cross-target reuse are not themselves new; Related Work already makes this distinction accurately.
- Points 2–4 are empirical contributions and insights. Say “tested setting/recipes” where relevant; do not turn finite comparisons into universal optimality claims.
- The 2.6% native advantage comes from a richer calibration recipe that jointly changes data, supervision, optimization, and runtime. Do not attribute it specifically to the two-term loss or extra data.
- The cross-family 14.3% end-to-end gain and unchanged Llama result make a useful Discussion takeaway: calibration benefits depend on the transfer pair; increased data/epochs are not automatically worthwhile. These are recipe-level comparisons, not isolated data-scaling effects.
- Native two-layer compression is worthwhile secondary evidence that input interfaces offer additional deployment control, but the four main points above form a tighter central story. Keep its quantitative evidence in Section 5.7 and the appendix.
- The contribution block should use LaTeX references, not fixed section/table numbers, and preferably start together on page 2 if keeping all four on page 1 would force typographic compression.

## Reader-facing answer to “where is it?”

After integration: end of **Introduction (pages 1–2)** for the explicit list; **Discussion (page 9)** for the combined interpretation; supporting evidence in **Section 4**, **Sections 5.7–5.9**, **Figure 2**, and **Tables 1, 3, 4, 6, 7**. Reconfirm page placement after rebuilding because line wrapping can shift.
