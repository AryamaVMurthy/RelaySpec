# Current manuscript claim–evidence map

8 September 2026. Main-paper throughput includes prefill except the adaptation table, explicitly labeled decode-only. All ratios remain inside their own runtime and request cohort.

| Main claim | Authoritative evidence | Boundaries |
|---|---|---|
| Frozen input-map fitting removes the source transformer for two inherited drafter families. | Source/native/relay generation implementation; primary `generated/evidence_registry.json` | Source embedding/head may remain. Linear connectors and adapter-only tuning have prior art. |
| Original Qwen maps reach 2.35–5.11× AR throughput. | `generated/ar_main_table.tex`, underlying paired raw records and core builder | 500 MATH requests per configuration, BF16, original 4,096-record fits. Not exact AR reproduction. |
| Original maps retain 89.1–90.3% native throughput; isolated EAGLE memory saves 7.63–7.80GiB. | `generated/native_retention_table.tex`, `resource_table.tex` | Three released native controls; memory measured separately on RTX6000Ada. |
| BF16 MATH accuracy differences are small but not established equivalent. | `generated/ar_quality_table.tex` | Exact sequence matches106–135/500; paired quality intervals include zero but do not prove noninferiority. |
| Fixed-work data scaling has diminishing returns. | `figures/expanded_numina_data.pdf` and registered raw evidence | 16–32,768 records,8,192updates,128 exposed development requests,one fittingseed. Not a universal512recordthreshold. |
| Frozen512maps retain97.5%/98.9% of2048map speed. | `generated/frozen_confirmation_table.tex`, frozen GSM8K confirmation records |256questions excluded from recorded developmenthistory. One-point accuracy noninferiority unresolved. |
| Dense maps lead measured8Bcapacity comparisons and feature regression leads matched warm-budget CE/LoRA controls. | Complete capacity tables/curves; `timed_budget_main_table.tex` and full budget evidence | Fixed training recipes. Adaptation screening16requests,256token cap. No optimum over all architectures/objectives. |
| Native-interface compression is a second application. | Native DFlash/EAGLE frozen GSM8K and code-capacity confirmation assets |60% fewerprojectionweights,97.1%/98.8% speed retention; code can require a thirdlayer. |
| Public RelaySpec configuration has higher measured throughput than tested PARD and frozen-drafterSD² configurations. | `public_baseline_comparison_table.tex`, public runtime registries | Different numerical/library/model contracts. Not algorithm-only superiority or a test of fullSD²/PARD2/TriSpec. |
| A richer linear recipe reaches native DFlash throughput with repeated2.6% advantage. | `rollout_transfer_table.tex`, `family_extension_evidence.json`, `transfer-reproduction-20260907/full16384-results` raw arrays/source/workerconfig |CustomNumina128requests,2048cap,batch-invariantBF16,vLLM0.28,onefit,two speculative runs,reusedAR. Data+loss+optimization+runtime change. |
| Llama and Qwen→Llama reuse reach2.52×/2.75× matchedAR with128/128 exactarrays. | `family128_table.tex`, `family_extension_evidence.json`,64sharedlane raw files |FP32target/map,BF16drafter,2048cap;128MATH excludedrecent40butnotgloballyunseen. Oneheterogeneoustokenizerpair. |
| Expanded cross-family recipe improves14.3%, while within-family improvement is unresolved. | Rebuilt paired128request intervals:[11.55,17.23]%cross,[-0.63,1.54]%Llama |Original4k differs from new8k/16k in recipe and work. Not data-only causality. |

No claim of universal exactness, best loss, minimal dataset size, optimized batched serving, semantic independence, general cross-tokenizer sampling, or superiority over all competitors is made. Larger generated-rollout fitting remains a positive separate operating point even though the same layer/context recipe did not win the Llama development search.

The main text prioritizes deployment value, calibration regimes and their limits. Full secondary studies remain in the appendix and raw experiment records. Current sources and citation status: `CITATIONS.md`, `citation-inventory.json`, `SOURCES-REVIEW.md`.
