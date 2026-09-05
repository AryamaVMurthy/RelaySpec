# EAGLE-3 optimized source-reuse profile

This four-GPU development profile uses the pinned DeepSpec implementation and
Qwen3-4B EAGLE-3 checkpoint, a Qwen3-8B verifier, 32 fixed MATH-500 prompts,
non-thinking chat formatting, greedy decoding, and a 2,048-token cap.

| Quantity | Result |
|---|---:|
| Native AR throughput | 44.26 token/s |
| Optimized source-reuse EAGLE throughput | 78.74 token/s |
| Source-reuse speedup over native AR | 1.779x |
| Source-trunk work | 102.324 s (33.33%) |
| Target-verification and draft work | 199.763 s (65.07%) |
| Other work | 4.897 s (1.60%) |
| Source-reuse committed tokens/cycle | 4.983 |
| Equal-acceptance, zero-overhead relay ceiling | 1.500x over source reuse |

The source path is already optimized: it executes only the committed prefix and
stops after zero-based layer 33, the final released EAGLE feature tap. Thus the
33.33% share is removable work in the strongest implemented source-reuse
baseline, not rejected-token or unused-layer inflation. This passes the
mechanism gate for fitting an EAGLE relay. Actual speed still depends on retained
acceptance and measured relay overhead, which the live relay probes evaluate.
