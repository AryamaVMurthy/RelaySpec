# Qwen3-8B RelaySpec source-trunk residency

Jobs 25475 and 25476 ran the same 32-prompt non-thinking MATH subset on
exactly four RTX 6000 Ada GPUs. Both loaded the Qwen3-4B source checkpoint to
retain its embedding and LM head; the unloaded condition then deleted the
unused transformer trunk and emptied the CUDA cache before measurement.

| Condition | Allocated before request | Peak allocated | Relay end-to-end tok/s | vs native AR |
|---|---:|---:|---:|---:|
| Source trunk resident | 23.92 GiB | 24.16 GiB | 210.22 | 4.670x |
| Source trunk unloaded | **17.10 GiB** | **17.34 GiB** | 209.66 | 4.668x |

Unloading the unused source trunk removes **6.82 GiB** of steady allocated GPU
memory (**28.5%**) and **6.82 GiB** of peak allocation (**28.2%**). Relay
throughput is 99.74% of the resident condition, so this storage/memory benefit
does not rely on keeping the source transformer in memory after relay loading.

The full raw artifacts, configs, allocation checks, source snapshots, and GPU
telemetry are in `reports/d8-memory-loaded/` and
`reports/d8-memory-unloaded/`.
