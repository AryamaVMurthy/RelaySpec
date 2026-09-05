# Isolated source-versus-relay memory residency

Separate processes evaluate `source_reuse_eagle3` and `relay_eagle3` on the same 8 requests.

| Metric | Source | Relay | Saved | Saved fraction |
|---|---:|---:|---:|---:|
| allocated before | 36.791 GiB | 29.369 GiB | 7.422 GiB | 20.17% |
| reserved before | 37.707 GiB | 30.039 GiB | 7.668 GiB | 20.34% |
| peak allocated | 37.565 GiB | 29.938 GiB | 7.627 GiB | 20.30% |
| peak reserved | 37.926 GiB | 30.219 GiB | 7.707 GiB | 20.32% |

The isolated micro-throughputs are contextual checks, not a paired speed estimate: source 70.522 tok/s, relay 77.726 tok/s.
