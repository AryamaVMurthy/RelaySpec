# Competitor and context campaign — 2026-09-06

## Completed: PARD-2

Job28201 completed in29m09s on four L40S GPUs. All384 rows (128questions × three methods) passed artifact checks; both speculative methods passed separate verifier-observed replica checks. Fixed2048-token cap, greedy generation.

| Method | tokens/s | speedup over matched AR | MATH accuracy | exact token matches to AR |
|---|---:|---:|---:|---:|
| native_ar | 31.09 | 1.00× | 80.47% | 128/128 |
| pard2_td | 135.08 | 4.34× | 79.69% | 32/128 |
| pard2_ti | 116.49 | 3.75× | 80.47% | 36/128 |

The accuracy difference95% paired intervals are[-3.91,+2.34] percentage points for TD and[-4.69,+4.69] for TI. Token outputs are not generally identical to AR, despite verifier-trace validity. Public runtime is pinned Transformers4.51.3/BF16/eager/static-cache; RelaySpec uses a different runtime. These results do not establish a runtime-independent superiority claim. These128 questions are development evidence.

## Context sweep

Exact input-token axis:4096/8192/16384/32768. Fixed256 generated-token cap,16 paired requests per full cell, DFlash and EAGLE-3 families. Each compares AR, native target-trained drafter, source reuse and the frozen dense N512 RelaySpec mapper. No new training. Report actual output lengths, request throughput, decode throughput, time to first token and peak/incremental memory. Contexts use natural mathematical background text and a fixed final task; this is controlled cost/retention evidence, not a standardized long-context benchmark.

Initial co-resident32K pilots28232/28233 failed with CUDA OOM; dependent jobs28245–28252 were canceled without allocation. Replacement pilots28256/28257 use separate processes per method, four workers at a time. DFlash RelaySpec unloads the unneeded source trunk. The baseline AR code retains an unused small DFlash drafter, which affects absolute memory. Method processes run in fixed sequence; no within-request order randomization is claimed.

Live job ledger:jobs.json; collected audits:run-JOBID/audit.json; collector:status.json. Only audited completed cells enter graphs. EDA feasibility is documented separately in EDA_FEASIBILITY.md; no substitute implementation is labeled EDA.

## Updated scaling status

DFlash32K pilot28256 passed all16 expected rows. DFlash4K full28258 passed64 rows; end-to-end throughput: AR34.21, native DFlash128.66, source reuse77.99, RelaySpec124.02 tokens/s (3.63× AR,96.4% of native). This is one16-question controlled-context cell, not a general ranking.

EAGLE isolated source-reuse32K pilot28257 failed with CUDA OOM:43.06 GiB allocated on a44.40 GiB device,608 MiB additional allocation requested. Record this as an implementation/hardware boundary, not a universal algorithmic limit. AR/native32K rows were completed before the failure. New pilot28266 tests AR/native/RelaySpec at32K; pilot28267 tests all methods at16K. Full EAGLE32K config explicitly excludes source reuse with the failing job and reason, and plots leave its cell absent.

Submitted full cells: DFlash28258/28259/28260/28261; EAGLE28268/28269/28270/28271, in4K/8K/16K/32K order. Dependencies serialize allocations and require successful relevant pilots. Collector exports four panels: end-to-end throughput, decode throughput, time to first token, peak allocated memory.
