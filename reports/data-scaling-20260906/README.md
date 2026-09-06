# Corrected experiment: distinct training examples

The user clarified that4K/8K/16K/32K means training records, not context tokens. Remaining context jobs28259–28261 and28268–28271 were canceled; no further context runs are authorized under this campaign. Completed PARD-2 comparisons remain valid and separate.

Four-GPU training job28272 fits one dense mapper per GPU at N4096/8192/16384/32768. All use8192 updates, four examples/update, seed1729, learning rate0.0006, no regularization, the same1024-record validation set and the already extracted, pinned NuminaMath feature cache. Fixed-update epochs are8/4/2/1. Saved1024/2048/4096/8192-update checkpoints also allow a one-epoch panel. The fit gate checks the actual distinct-record count.

This is a separately labeled NuminaMath data curve, not an unlabeled extension of the older MATH-only curve. A1024-update run cannot expose more than4096 records, so that budget cannot represent32768 distinct training examples.

Evaluation pilot28273 follows successful fitting; full evaluation28274 follows the pilot. The full run uses128 paired development questions,2048 generated-token cap and shared AR/native-DFlash/source-reuse references. It also re-evaluates the existing same-protocol Numina N512/N2048 fixed8192-update checkpoints. Seven new checkpoints cover both panels because N32768 fixed-work and one-epoch are the same fit.

The collector checks completion gates and runs the pinned math scorer. scripts/report_record_scaling.py produces results.json and distinct-record-scaling.png/pdf after completion, including quality, paired uncertainty and fitting diagnostics. These are single-seed development results. No new data-scaling result is claimed before the jobs complete.

See jobs.json for exact commands and immutable snapshots; collector-status.json for observed progress.
