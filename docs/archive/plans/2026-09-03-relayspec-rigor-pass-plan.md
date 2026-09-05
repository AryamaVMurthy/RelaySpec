# Plan: ICLR-rigor pass on RelaySpec

Date 2026-09-03. Direct response to explicit user requirements (verbatim
list preserved below), scoped against what turing's real 8-GPU shared
quota can execute in this engagement.

## Triage

**Tractable, executing now:**
1. Reposition novelty framing (writing).
2. Cheap-adaptation baselines: closed-form ridge regression, proposer
   input-layer-only fine-tune, LoRA proposer tuning, CE/KL adaptation
   (already have L1/L2 from the source-free line, needs correct framing).
3. Linear-vs-MLP relay comparison, data-size scaling study. Tap-layer
   count ablation is architecturally blocked (the frozen proposer's own
   checkpoint fixes the tap count at 5; changing it needs a different
   proposer, already established in the E5 scope decision), stated
   honestly rather than silently dropped.
4. Expand transfer evidence: more prompts, bootstrap CIs, both
   directions, second family.
6. Reframe cost-benefit (89-91%) as the headline (writing).
7. Elevate break-even equation as a central contribution (writing).
8. Fix SD² description and related-work positioning (research + writing).
9. Strengthen the representation-level finding (writing, synthesizing
   already-measured results).
10. Reporting fixes: prompt-count consistency, CIs, matmul overhead,
    keep the exact-output audit.

**Not completable this session, scoped explicitly:**
5. vLLM/SGLang systems evaluation. Requires either forking vLLM/SGLang's
   scheduler to route through a custom relay-conditioned proposer path,
   or writing a new engine backend. This is a separate, multi-day
   project. Documented as a stated limitation with the honest reason
   (already partially done via the SPEED-Bench citation), not attempted
   as a partial/misleading gesture.

## Execution order

1. Implement closed-form ridge regression, input-layer-only fine-tune,
   LoRA proposer tuning, and MLP relay variant as new training modes
   (code, local, testable without GPU).
2. Launch data-size scaling sweep and expanded-prompt transfer runs
   across turing's free nodes in parallel (node02/03/04/07/09 confirmed
   idle).
3. Compute relay matmul overhead explicitly from existing profiling
   JSON (no new run needed, already recorded).
4. Rewrite paper sections for framing/positioning/synthesis once results
   are in.
