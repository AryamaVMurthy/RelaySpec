# EDA feasibility check

Reviewed official repository commit `a29f0381479a92d0f3114fd264cd790bab3a208b`, downloaded read-only to `/tmp/relayspec-eda-review-20260906`.

The released scripts train a stage-1 shared/private EDA architecture and require its checkpoint for stage-2 transfer. Examples use Qwen2.5-7B to Qwen2.5-Math/Coder/medical variants. The repository does not provide a directly usable checkpoint path in those scripts; paths are user-filled placeholders. This is not an input-map adaptation of the available DFlash/EAGLE-3 checkpoints.

A faithful run needs compatible base/domain target models, extracted target features, a valid EDA stage-1 checkpoint or its training, and a RelaySpec comparison on that same transfer setting. Our Qwen3 4B-to-8B/14B mapper experiment is not a drop-in substitute. EDA remains not run; this feasibility finding must be retained alongside the queued PARD-2 and context studies. No GPU job was submitted under the EDA label.
