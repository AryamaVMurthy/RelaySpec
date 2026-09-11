These modules were copied from the validated Turing ZIP reproduction package
`/scratch/node07/aryama.murthy/transfer-reproduction-20260907/package/src`
on 2026-09-11. They provide the original mapper registration, source embedding
and head checks, GPU telemetry, and measured generation loop. Their source
snapshot is tracked in this branch; original reproduction outputs stay read-only.

The AUF study prepares a separate `TRANSFER_WORK` for each checkpoint. Its
benchmark manifests identify development versus confirmation data explicitly.
The four-request, 128-token pilot is diagnostic only; main reported evaluations
use 128 distinct requests and a 2,048-token cap with natural EOS.
