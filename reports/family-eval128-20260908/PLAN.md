# Frozen 128-question evaluation

Compare the old and selected dense mapper for Llama→Llama and Qwen→Llama against matched FP32 autoregressive decoding. The output cap is **2,048 new tokens**, with ordinary EOS stopping. Drafter stays BF16; selected blocks stay 10 and 16. No tuning on these results.

The same 128 MATH500 records are used for both pairs: shuffle the pinned repository MATH500 records with Python seed 1729 and take indices 40:168. These exclude the 40 questions used in recent family pilot/development/final checks. This is not a claim that the records were never used elsewhere in the broader project.

Array 28982 has sixteen sequential four-GPU waves on node07. Each GPU loads one pair and evaluates AR, old mapper, and selected mapper on four records. Two GPUs serve each family, with disjoint shards. Shared AR and method rotation follow the previously validated shared-candidate runner. Each method receives exactly 128 unique requests at completion.

Full generated token ID sequences are saved after timing, along with SHA256 hashes. Aggregation validates hashes and lengths, checks unique complete coverage, and compares token arrays directly. Every disagreement is retained with its first differing position. Exact match here means agreement with the matched FP32 AR sequence, not mathematical answer accuracy or a universal guarantee.

Results are collected under artifacts/ and status.json. Old/new throughput is interpretable only after all requests finish; incomplete method cohorts can differ. No retraining is performed in this evaluation.
