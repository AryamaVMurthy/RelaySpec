# Isolated source-versus-relay memory residency

Separate processes evaluate `source_reuse_eagle3` and `relay_eagle3` on the same 8 requests.

| Metric | Source | Relay | Saved | Saved fraction |
|---|---:|---:|---:|---:|
| allocated before | 24.541 GiB | 17.093 GiB | 7.448 GiB | 30.35% |
| reserved before | 25.908 GiB | 18.035 GiB | 7.873 GiB | 30.39% |
| peak allocated | 25.563 GiB | 17.763 GiB | 7.800 GiB | 30.51% |
| peak reserved | 26.141 GiB | 18.195 GiB | 7.945 GiB | 30.39% |

The isolated micro-throughputs are contextual checks, not a paired speed estimate: source 82.816 tok/s, relay 97.169 tok/s.
