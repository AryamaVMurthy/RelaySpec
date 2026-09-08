# Data, capacity, and optimization visual evidence

Completed two reproducible figure additions without new GPU runs.

## Main paper: `figures/visual_scaling_main.pdf`

The 6.2 × 2.75 inch, two-panel figure replaces the oversized, single-panel data plot. All labels are 8–9.4 pt at native size. Blue, green, and orange are distinguished by marker shapes and line styles.

- **(a) Training records:** all eleven 16–32,768-record NuminaMath cells measured together at 8,192 batch-four updates. The curve shows the rapid improvement and later flattening directly; 512 records retain 97.2% of the observed best throughput.
- **(b) Mapper capacity:** the seven 8,192-update candidates in the completed 128-question, 2,048-output-token DFlash-8B comparison. It shows parameter count against downstream throughput, including both dense checkpoints, factored linear widths 512/1,024/4,096, and MLP widths 512/4,096. Open symbols indicate 512 records, filled symbols 2,048. Only the two 2,048-record width endpoints are connected within each nondense family. This is the completed long-output subset, not all 30 fitting-only capacity cells.

Suggested caption:

> DFlash-8B calibration on 128 exposed MATH development questions, with a 2,048-output-token cap. (a) All eleven record counts share controls in one fixed-update campaign. (b) The seven standard-update candidates in a separate long-output capacity campaign; open symbols use 512 records and filled symbols 2,048. Error bars are paired 95% request intervals conditional on one fitting seed; they exclude fitting-seed, repeated-run, and selection uncertainty. Throughput includes prefill.

Both axes report end-to-end throughput divided by the matched AR throughput **within their own campaign**. The panels do not share timings or an AR denominator. These inherited BF16 development results do not claim exact token agreement with AR.

## Appendix: `figures/visual_scaling_loss_speed.pdf`

The 6.2 × 2.9 inch diagnostic plots train and validation relative interface MSE against end-to-end throughput for the same seven candidates plus two longer-training endpoints. The dense-2k and MLP-4096-2k annotations make their training-fit/throughput ordering visible. Arrows and × symbols show width-512 factored/MLP continuation from 8,192 to 32,768 updates: better feature fitting produces only modest throughput changes. Open markers again use 512 records; all others use 2,048.

Suggested caption:

> Better feature fitting need not yield the fastest decoder. The nine evaluated DFlash-8B mapper recipes share 128 exposed MATH development requests and a 2,048-token cap. Dense-2k is faster than MLP-4096-2k despite the latter's lower training loss. Arrows lead from 8,192 to 32,768 updates for width-512 factored and MLP maps. These are descriptive recipe comparisons, not a causal model of decoding speed; lower validation loss generally accompanies higher throughput within this selected set.

Do not claim no association between validation loss and speed. The numerical registry records the descriptive correlations for audit, but neither a regression fit nor significance claim is plotted. Architecture, parameter count, record count, and repeated checkpoint observations are not independent interventions in this scatter.

## Existing figures retained

The existing `numina_capacity`, `numina_epochs`, `numina_regularization`, `target14_dflash_fitting`, `target14_eagle3_fitting`, and `eagle_small_epochs` figures already display the full fit matrix, continued trajectories, regularization and additional-family fitting curves. Redrawing them would duplicate evidence and add little analytical coverage.

## Reproduction and validation

```bash
/home/aryamavmurthy/work/RelaySpec/.venv/bin/python scripts/build_scaling_visuals.py --root . --output paper/iclr2027
```

`build(root, output)` is also callable by a master manuscript builder. It emits both PDF and PNG, plus `generated/scaling_visuals_evidence.json` with all plotted numerical values and SHA256 digests of every consumed file (47 inputs at completion). The sibling `RelaySpec` checkout is a read-only fallback for original archived raw capacity evidence, consistent with existing manuscript builders; resolved paths are explicit in the registry.

Checks completed:

- Recomputed throughput from raw total output-token and request-time sums for every method in both campaigns.
- Verified exactly 128 distinct, matched request identities per method, with no duplicated or omitted requests.
- Checked the generation cap, greedy setting, target, family and configured 128-request budget from the raw run configurations.
- Checked all eleven record counts against actual `distinct_records_seen` and 8,192-update fitting records.
- Validated capacity and continuation training/validation losses against raw trajectories and frozen SHA256 provenance.
- Opened and inspected both generated PNGs for legibility, overlap, clipping, labels and legends.
- Rebuilt into a temporary directory and verified byte-for-byte identical PDFs, PNGs and evidence registry. PDF timestamps are suppressed.

No manuscript text or shared figure builders were edited by this subtask.
