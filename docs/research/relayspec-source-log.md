# RelaySpec focused source log

Checked 2026-08-28. This log records only sources used to design the ICLR
evaluation. Preprints are explicitly distinguished from peer-reviewed work.

## Relay-fit data

- **Dataset:** `DigitalLearningGmbH/MATH-lighteval`, train split, immutable
  revision `0530c78699ea5e8eb5530600900e1f328b48acad`.
- **Use:** a deterministic 4,096-record subset supplies ordinary non-thinking
  math text for interface regression only. Target, source, and proposer weights
  remain frozen; answers are not used as a task-loss target.
- **Integrity:** `configs/train_math_4096.json` records the source, revision,
  per-problem hashes, selection seed, and manifest digest. The
  operator-preserving normalized-hash audit reports zero exact overlap with all
  1,250 MATH/GSM8K/HumanEval/MBPP/MT-Bench evaluation records in
  `reports/final/FIT_EVALUATION_OVERLAP_AUDIT.json`.

## DFlash

- **Title:** DFlash: Block Diffusion for Flash Speculative Decoding
- **Authors:** Jian Chen, Yesheng Liang, Zhijian Liu
- **Status:** accepted at ICML 2026; camera-ready v2 dated 2026-05-28
- **URLs:** camera-ready https://arxiv.org/abs/2602.06036; official ICML
  program https://icml.cc/Downloads/2026
- **Bibliographic caution:** the ICML program confirms acceptance, but the PMLR
  proceedings volume was not published at the time of this audit. No volume or
  page range is inferred.
- **Relevance:** Primary proposer used by RelaySpec. It reports greedy and
  lossless-sampling results on GSM8K, MATH-500, AIME25, HumanEval, MBPP,
  LiveCodeBench, and MT-Bench; uses a 2,048-token generation cap; reports
  acceptance length and end-to-end speedup; and includes SGLang concurrency
  measurements. Its training set contains about 800K target-generated samples.
- **Checked implementation locators:** DFlash commit
  `94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756` fixes the benchmark CLI default
  at 2,048 generated tokens in `dflash/benchmark.py:485`; its README examples
  use block 16 and 2,048 tokens. The camera-ready method section specifies five
  draft layers, block 16 for Qwen, and five target depths uniformly spanning
  the second through third-to-last layers. These details support comparability
  settings only, not RelaySpec performance.

## Canonical speculative decoding

- **Title:** Fast Inference from Transformers via Speculative Decoding
- **Authors:** Yaniv Leviathan, Matan Kalman, Yossi Matias
- **Status:** ICML 2023, PMLR 202:19274--19286, peer reviewed
- **URL:** https://proceedings.mlr.press/v202/leviathan23a.html
- **Relevance:** Primary source for the rejection/correction construction that
  preserves the target distribution. It supports the formal algorithm, not the
  correctness of RelaySpec's cache implementation; executable conformance tests
  are still required.

## DFlare

- **Title:** DFlare: Scaling Up Draft Capacity for Block Diffusion Speculative
  Decoding
- **Authors:** Jiebin Zhang et al.
- **Status:** arXiv preprint, 2026
- **URL:** https://arxiv.org/abs/2606.02091
- **Relevance:** Strong post-DFlash baseline and evidence that per-layer target
  feature use affects acceptance. It evaluates Qwen3-4B, Qwen3-8B, and
  GPT-OSS-20B across math, code, and chat, and reports both speedup and
  acceptance. It trains on 2.4M samples and reports large training cost, making
  adaptation cost a necessary RelaySpec metric.

## KVShot

- **Title:** When Hidden States Drift: Can KV Caches Rescue Long-Range
  Speculative Decoding?
- **Authors:** Tianyu Liu, Yuhao Shen, Xinyi Hu, Baolin Zhang, Hengxin Zhang,
  Jun Dai, Jun Zhang, Shuang Ge, Lei Chen, Yue Li, MingCheng Wan
- **Status:** arXiv preprint, v2 dated 2026-05-09
- **URL:** https://arxiv.org/abs/2604.26412
- **Relevance:** Directly tests hidden-only, target-KV-only, and hybrid reuse
  for Qwen3-8B drafting. It finds improved long-range acceptance from target
  KV reuse but only marginal end-to-end speedup under its training pipeline.
  This narrows RelaySpec's claim: verifier-state reuse itself is not new. The
  distinct systems contribution is a small per-target translator that emulates
  the conditioning interface of an already-trained frozen proposer, removes a
  separately executed source transformer, and demonstrates positive measured
  end-to-end speed and memory effects.

## xPress

- **Title:** xPress: Parallel Refinement for Diffusion Drafters in Speculative
  Decoding
- **Authors:** Zheng Wang et al.
- **Status:** arXiv preprint, 2026
- **URL:** https://arxiv.org/abs/2608.02438
- **Relevance:** Latest acceptance-improvement reference for DFlash. It uses
  Qwen3-8B non-thinking mode, target-regenerated Open-PerfectBlend data, the
  standard math/code/chat suite, acceptance length, and end-to-end throughput.
  Its results reinforce that RelaySpec must report the acceptance loss caused
  by its translated interface rather than only component latency.

## CaDDTree

- **Title:** Cost-Aware Diffusion Draft Trees for Speculative Decoding
- **Authors:** Shuai Zhang, Huachuan Qiu, Hongliang He, Yong Dai
- **Status:** arXiv preprint, 2026
- **URL:** https://arxiv.org/abs/2606.01813
- **Relevance:** Establishes throughput rather than acceptance alone as the
  optimization target. It profiles verification cost over context length,
  uses repeated warm timings, evaluates greedy and sampling, and reports eight
  math/code/chat benchmarks with a 2,048-token cap.

## Distributed Speculative Inference

- **Title:** Distributed Speculative Inference: Speculation Parallelism for
  Provably Faster Lossless Language Model Inference
- **Authors:** Nadav Timor et al.
- **Status:** ICLR 2025 poster
- **URL:** https://openreview.net/forum?id=cJd1BgZ9CS
- **Relevance:** Peer-reviewed reference for lossless speculative inference and
  for separating algorithmic exactness from empirical token matches.

## Hierarchical Speculative Decoding

- **Title:** Overcoming Joint Intractability with Lossless Hierarchical
  Speculative Decoding
- **Authors:** Yuxuan Zhou, Fei Huang, Heng Li, Fengyi Wu, Tianyu Wang, Jianwei
  Zhang, Junyang Lin, Zhi-Qi Cheng
- **Status:** ICLR 2026 oral
- **URL:** https://openreview.net/pdf/30c4808dce409c1c13f400191c77ebc2dc966531.pdf
- **Relevance:** A current top-venue standard for a speculative-decoding paper:
  it proves distributional correctness, evaluates multiple model families and
  benchmarks, sweeps temperature and draft length, verifies task quality, and
  demonstrates integration with EAGLE-3. RelaySpec should match this separation
  between theorem-backed exactness and empirical systems gains.

## EAGLE-3

- **Title:** EAGLE-3: Scaling up Inference Acceleration of Large Language Models
  via Training-Time Test
- **Authors:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang
- **Status:** NeurIPS 2025 main conference
- **URL:** https://proceedings.neurips.cc/paper_files/paper/2025/hash/c7b5a35ea98b62512a869c19ea7b03cb-Abstract-Conference.html
- **Relevance:** Evaluates four target models, five tasks, greedy and
  temperature-one decoding, speedup and acceptance length, component
  ablations, data scaling, and SGLang/vLLM batch sweeps. This is the closest
  peer-reviewed template for RelaySpec's experiment structure. Page 6 reports
  approximately 68K ShareGPT plus 464K UltraChat entries with target-generated
  responses; this directly grounds the adaptation-data comparison, while its
  separate 16-A100/two-week statement applies only to the 70B head and is not
  extrapolated to our 8B/14B setting.

## GRIFFIN

- **Title:** GRIFFIN: Effective Token Alignment for Faster Speculative Decoding
- **Authors:** Shijing Hu, Jingyang Li, Xingyu Xie, Zhihui Lu, Kim-Chuan Toh,
  Pan Zhou
- **Status:** NeurIPS 2025 main conference
- **URL:** https://proceedings.neurips.cc/paper_files/paper/2025/hash/b6e67ae290635d0874c4cb43ba2a2cfb-Abstract-Conference.html
- **Relevance:** Evaluates LLaMA, Vicuna, Qwen, and Mixtral on dialogue, code,
  and math and diagnoses alignment degradation across drafting steps. RelaySpec
  should analogously report relay error and acceptance at each block position,
  not only mean acceptance.

## MoESD

- **Title:** MoESD: Unveil Speculative Decoding's Potential for Accelerating
  Sparse MoE
- **Authors:** Zongle Huang, Lei Zhu, ZongYuan Zhan, Ting Hu, Weikai Mao,
  Xianzhi Yu, Yongpan Liu, Tianyu Zhang
- **Status:** NeurIPS 2025 main conference
- **URL:** https://proceedings.neurips.cc/paper_files/paper/2025/hash/b637af7745d3ad4cb0b9cdaa056ab41e-Abstract-Conference.html
- **Relevance:** Shows that acceptance alone does not determine speed and
  validates a performance model over batch size, architecture, and different
  GPUs. It motivates RelaySpec's Amdahl decomposition and concurrency sweep.

## Medusa

- **Title:** Medusa: Simple LLM Inference Acceleration Framework with Multiple
  Decoding Heads
- **Authors:** Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee,
  Deming Chen, Tri Dao
- **Status:** ICML 2024
- **URL:** https://proceedings.mlr.press/v235/cai24b.html
- **Relevance:** Separates a frozen-backbone lossless method from a jointly
  tuned higher-speed method and reports parameter, training, and quality-speed
  tradeoffs. RelaySpec should be equally explicit that its target and proposer
  remain frozen and that its adapter is the only trainable component.

## Heterogeneous-vocabulary speculative decoding

- **Title:** Accelerating LLM Inference with Lossless Speculative Decoding
  Algorithms for Heterogeneous Vocabularies
- **Authors:** Nadav Timor, Jonathan Mamou, Daniel Korat, Moshe Berchansky,
  Gaurav Jain, Oren Pereg, Moshe Wasserblat, David Harel
- **Status:** ICML 2025
- **URL:** https://proceedings.mlr.press/v267/timor25a.html
- **Relevance:** A peer-reviewed example of broadening proposer reuse while
  preserving the target distribution. It evaluates summarization, code, and
  long-context workloads and reports wall-clock speed rather than proxy metrics
  alone. RelaySpec remains narrower: compatible vocabulary and model family.

## Variational Speculative Decoding

- **Title:** Variational Speculative Decoding: Rethinking Draft Training from
  Token Likelihood to Sequence Acceptance
- **Authors:** Xiandong Zou, Jianshu Li, Jing Huang, Pan Zhou
- **Status:** arXiv preprint, 2026
- **URL:** https://arxiv.org/abs/2602.05774
- **Relevance:** Closest prior work to sequence-acceptance training. It uses a
  variational path objective and EM/MCMC training. Therefore RelaySpec must not
  claim that optimizing expected accepted length or prefix survival is new; a
  defensible contribution is a lightweight survival-aligned surrogate
  specialized to frozen cross-target interface adaptation, if it wins the
  objective ablation.

## PARD

- **Title:** PARD: Accelerating LLM Inference with Low-Cost PARallel Draft
  Model Adaptation
- **Authors:** Zihao An, Huajun Bai, Ziqiong Liu, Dong Li, Emad Barsoum
- **Status:** ICLR 2026
- **URL:** https://openreview.net/forum?id=XbOyv7iVGL
- **Relevance:** Closest portability framing: one target-independent parallel
  drafter can serve a model family. PARD designs and trains its drafter for that
  property; RelaySpec instead translates a new target's hidden states into the
  interface of an already-trained, frozen target-specific proposer.

## Steering Pretrained Drafters during Speculative Decoding

- **Title:** Steering Pretrained Drafters during Speculative Decoding
- **Authors:** Frédéric Berdoz, Peer Rheinboldt, Roger Wattenhofer
- **Status:** AAAI 2026
- **URL:** https://ojs.aaai.org/index.php/AAAI/article/view/40255
- **Relevance:** This is the closest conceptual prior work. SD-squared computes
  a steering vector from verifier hidden states and injects it into a pretrained
  autoregressive drafter, reporting acceptance gains of up to 35% under
  sampling and 22% under greedy decoding. RelaySpec must not claim verifier
  feature steering as new. Its narrower difference is to emulate the existing
  learned conditioning interface of a feature-conditioned DFlash/EAGLE
  proposer while leaving that proposer structurally unchanged and frozen, for
  the systems purpose of removing a repeated source trunk.

## OmniDraft

- **Title:** OmniDraft: A cross-vocabulary, online adaptive drafter for
  on-device speculative decoding
- **Authors:** Ramchalam Kinattinkara Ramakrishnan, Zhaocong Yuan, Shaojie
  Zhuo, Chen Feng, Yicheng Lin, Chenzheng Su, Xiaopeng Zhang
- **Status:** NeurIPS 2025 main conference
- **URL:** https://proceedings.neurips.cc/paper_files/paper/2025/hash/3c2fe1417eed1c6ff9acf169617981ea-Abstract-Conference.html
- **Relevance:** Demonstrates one generic 68M drafter across Vicuna, Qwen2, and
  Llama3 with online adaptation and cross-vocabulary handling. RelaySpec must
  not use a broad “one drafter for all targets” novelty claim; it is a
  per-target translator for reusing an already trained feature-conditioned
  proposer within a compatible interface family.

## PARD-2

- **Title:** PARD-2: Target-Aligned Parallel Draft Model for Dual-Mode
  Speculative Decoding
- **Authors:** Zihao An, Taichi Liu, Ziqiong Liu, Dong Li, Ruofeng Liu, Emad
  Barsoum
- **Status:** arXiv preprint, 2026-05-09
- **URL:** https://arxiv.org/abs/2605.08632
- **Relevance:** Directly optimizes consecutive acceptance with adaptive token
  reweighting and supports target-dependent and target-independent modes. This
  prevents RelaySpec from claiming acceptance-aware training in general. The
  prefix-survival loss is only a lightweight refinement specialized to a
  frozen cross-target interface adapter and must win an objective ablation.

## Efficient Draft Adaptation

- **Title:** Efficiently Aligning Draft Models via Parameter- and Data-Efficient
  Adaptation
- **Authors:** Luxi Lin, Zhihang Lin, Zhanpeng Zeng, Yuhao Chen, Qingyu Zhang,
  Jixiang Luo, Xuelong Li, Rongrong Ji
- **Status:** arXiv preprint, 2026-03-10
- **URL:** https://arxiv.org/abs/2603.09527
- **Relevance:** EDA adapts draft models to fine-tuned targets using a
  decoupled shared/private architecture, target-regenerated data, and sample
  selection. It is a necessary efficiency baseline and further narrows
  RelaySpec's claim to frozen learned-interface emulation rather than generic
  parameter-efficient draft adaptation.

## MetaSD

- **Title:** Multi-Drafter Speculative Decoding with Alignment Feedback
- **Authors:** Taehyeon Kim, Hojung Jung, Se-Young Yun
- **Status:** ACL 2026 Findings
- **URL:** https://arxiv.org/abs/2604.05417
- **Relevance:** Dynamically allocates work to heterogeneous drafters with a
  multi-armed bandit. It shows that using several drafters is not independently
  novel. RelaySpec therefore trains three seeds only to measure oracle union
  coverage and activates a multi-relay extension only after a cost gate.

## vLLM DFlash LoRA RFC

- **Title:** LoRA adapter support for DFlash speculative decoding draft models
- **Author:** Ansh Kaggarwal
- **Status:** open implementation RFC, 2026-08-12; not peer reviewed
- **URL:** https://github.com/vllm-project/vllm/issues/52038
- **Relevance:** Proposes per-domain LoRA adapters for a shared DFlash drafter
  and reports an R64 adapter about 28 times smaller than a full 0.8B drafter,
  with acceptance behavior within about 2% of a full domain-specific drafter.
  It does not translate a new target's hidden state into a frozen proposer's
  old target interface, but it means RelaySpec cannot claim small DFlash
  adapters broadly as unexplored.

## DART

- **Title:** DART: Diffusion-Inspired Speculative Decoding for Fast LLM
  Inference
- **Authors:** Fuliang Liu, Xue Li, Ketai Zhao, Yinxi Gao, Ziyan Zhou,
  Zhonghui Zhang, Zhibin Wang, Wanchun Dou, Sheng Zhong, Chen Tian
- **Status:** arXiv preprint, 2026
- **URL:** https://arxiv.org/abs/2601.19278
- **Relevance:** Establishes that target hidden states feeding a lightweight
  parallel future predictor is prior art. RelaySpec cannot claim hidden-feature
  conditioning or parallel prediction as its novelty.

## Multi-Candidate Speculative Decoding

- **Title:** Multi-Candidate Speculative Decoding
- **Authors:** Sen Yang, Shujian Huang, Xinyu Dai, Jiajun Chen
- **Status:** arXiv preprint, 2024
- **URL:** https://arxiv.org/abs/2401.06706
- **Relevance:** Samples and verifies multiple candidates while preserving the
  target distribution. Consequently, merely adding multiple relay outputs or
  majority voting is not a sufficient contribution; a multi-relay extension
  must be justified by cross-target representation ambiguity and whole-cycle
  cost, and consensus may rank candidates but cannot replace target
  verification.

## DARTree

- **Title:** DARTree: Speculative Diffusion Decoding with Autoregressive Draft
  Trees
- **Authors:** Tianyi Li, Yaxin Luo, Xinyi Shang, Zhiqiang Shen
- **Status:** arXiv preprint, 2026-08
- **URL:** https://arxiv.org/abs/2608.13524
- **Relevance:** Current diffusion-tree baseline. It shows that broadening
  diffusion proposals into a verification tree is already active work. Any
  RelaySpec ensemble must compare against one-proposer multi-candidate/tree
  generation under the same node and latency budget.

## Speculative KV Coding

- **Title:** Speculative KV coding: losslessly compressing KV cache by up to
  approximately 4x using a predictor model
- **Author:** Fergus Finn
- **Status:** Technical note, not peer reviewed, 2026-05-08
- **URL:** https://resources.doubleword.ai/resources/speculative-kv-coding-losslessly-compressing-kv-cache-by-up-to-4x-using-a-predictor-model
- **Relevance:** Adjacent representation-prediction idea. It is not evidence
  for RelaySpec performance and must not be presented as a peer-reviewed
  baseline.

## Cross-Model KV Cache Transfer

- **Title:** Cross-Model KV Cache Transfer in LLM Families: A Closed-Form
  Linear Mapping for Prefill Reuse
- **Authors:** Taekyung Heo, Rasoul Shafipour, Ritchie Zhao, Maximilian Golub,
  Mohammad Mahdi Kamani, Ritika Borkar, Makesh Tarun Chandran, Pantea
  Zardoshti, Bita Darvish Rouhani
- **Status:** arXiv preprint, 2026-08-04
- **URL:** https://arxiv.org/abs/2608.03893
- **Checked claims:** Maps source-model KV into receiver-model KV to skip a
  receiver prefill. The production mapper is per-head ridge regression with
  cross-layer selection and RoPE-stripped keys; reported mapper sizes are
  1.01--3.36B parameters (4--12 GB), fit in roughly 47--87 minutes on eight
  H100s. It reports 73--98% standalone-accuracy retention on the better pairs
  and 2.7--25x mapper versus re-prefill latency.
- **Relevance and boundary:** This is the closest current representation-
  translation neighbor, so RelaySpec cannot claim that cross-model linear
  mapping is new. Its intervention and guarantee differ: it predicts the
  receiver's per-layer KV to bypass receiver prefill and can lose downstream
  accuracy; RelaySpec maps already-mandatory target taps into one frozen
  proposer's conditioning interface during decode, removes the separate source
  trunk, and leaves full-target verification as sole token authority. RelaySpec
  also demonstrates the same interface-level intervention across diffusion and
  autoregressive-chain proposers rather than matched-KV model-family handoff.

## Verifier Skipping

- **Title:** From Positionwise Confidence to Prefix Scheduling: Verifier
  Skipping in Speculative Decoding
- **Authors:** Haoxuan Luo, Jameson Sandler, Ferdinando Fioretto
- **Status:** arXiv preprint, 2026-08-14
- **URL:** https://arxiv.org/abs/2608.14787
- **Relevance and boundary:** This August 2026 work accelerates the dominant
  verifier component by sometimes committing draft prefixes without invoking
  the verifier, explicitly introducing a lossy policy axis. It is important
  evidence that verifier calls remain a separate major bottleneck, but it is
  not a substitute for RelaySpec: RelaySpec removes the redundant source trunk
  while invoking the full verifier on every cycle and preserving target-only
  token authority. Its reported HumanEval result concerns verifier-call savings
  at matched observed pass@1, not distributional exactness.

## SpecSA

- **Title:** SpecSA: Bridging Speculative Decoding and Sparse Attention for
  Efficient LLM Inference
- **Authors:** Zhibin Wang, Ziyu Zhong, Nuo Shen, Yuhang Zhou, Rong Gu, Sheng
  Zhong
- **Status:** arXiv preprint, 2026-05-19
- **URL:** https://arxiv.org/abs/2605.19893
- **Relevance and boundary:** SpecSA reduces the target verification cost for
  long-context sparse-attention models through verification-oriented kernels
  and orchestration. It is orthogonal to interface transplantation and supports
  the paper's Amdahl framing: after eliminating the source trunk, verifier
  kernels are the remaining dominant component. No SpecSA speed number is used
  to justify a RelaySpec constant or extrapolate the current dense-SDPA runs.

## HyperDFlash

- **Title:** HyperDFlash: Hyper-Connection-Aligned Block Speculative Decoding
  with Gated Residual Reduction
- **Authors:** Luxi Lin, Shuang Peng, Rui Ma, Junhao Hua, Shuwei Fan, Zhengda
  Qin, Qiang Wang, Hongjian Sun, Fangmin Chen, Songwei Liu
- **Status:** arXiv preprint v2, 2026-06-29; not treated as peer reviewed
- **URL:** https://arxiv.org/abs/2606.26744
- **Checked claims:** HyperDFlash trains a new six-token DFlash for one
  DeepSeek-V4-Flash target. It conditions on that target's final pre-collapse
  hyper-connection residual, inherits its input-dependent `hc_head` gate for
  path reduction, and trains on roughly 300K general plus 150K task-oriented
  examples for two five-epoch stages on eight H20 GPUs. Its primary metrics
  are speedup over target AR and mean accepted length across math, code, and
  chat. The paper itself notes that a production component/latency study is
  still missing.
- **Relevance and boundary:** This is the closest current evidence that the
  exact architectural conditioning boundary matters and that redundant
  intermediate-feature computation should be measured. It blocks any broad
  claim that target-interface alignment for DFlash is new. RelaySpec differs
  in the narrow intervention: it keeps an already-trained source proposer
  frozen, learns only a target-to-old-interface translator, and removes the
  separately executed source transformer during every speculative cycle. Its
  DFlash/EAGLE cross-family result, source-reuse denominator, component timing,
  and memory release are therefore the required differentiators.

## AngelSpec

- **Title:** AngelSpec: Towards Real-World High Performance Inference with
  Speculative Decoding
- **Authors:** Hong Liu et al.
- **Status:** arXiv preprint, 2026-07-28; not treated as peer reviewed
- **URL:** https://arxiv.org/abs/2607.25852
- **Checked claims:** AngelSpec co-specializes an autoregressive MTP drafter and
  a newly trained block-diffusion DFly drafter by workload. DFly changes the
  proposal architecture, combines target conditioning with a
  predecessor-conditioned head, and reallocates verification depth online
  using a profiled utility model. It reports concurrency-aware throughput on
  the Hy3 series.
- **Relevance and boundary:** This is current evidence that proposer structure,
  task distribution, acceptance, verification cost, and load jointly determine
  real throughput. It does not reuse a frozen proposer trained for another
  target or eliminate the old target's separately executed source trunk.
  RelaySpec therefore retains its narrow transplantation contribution, while
  its batch-one dense-SDPA results must not be presented as a serving-throughput
  comparison to AngelSpec.

## RepSpec

- **Title:** RepSpec: Structural Re-parameterized Draft Model Training for
  Speculative Decoding
- **Status:** ICLR 2026 conference paper; peer reviewed
- **URL:** https://proceedings.iclr.cc/paper_files/paper/2026/hash/d70ea003729b440b89a2f958a5554c1f-Abstract-Conference.html
- **Checked claims:** RepSpec adds redundant linear/nonlinear structures while
  training an EAGLE drafter and merges them into its backbone for inference,
  improving accepted length without adding inference structure.
- **Relevance and boundary:** RepSpec supports reporting accepted length and
  separating training-only capacity from inference cost. It improves a newly
  trained target-specific drafter; it does not keep an old proposer unchanged,
  emulate its conditioning interface, or eliminate a repeated source trunk.

## SPEED-Bench

- **Title:** SPEED-Bench: A Unified and Diverse Benchmark for Speculative
  Decoding
- **Status:** ICML 2026 accepted paper; official program and arXiv:2604.09557
  checked
- **URL:** https://arxiv.org/abs/2604.09557
- **Checked claims:** speculative-decoding behavior is data- and
  concurrency-dependent. SPEED-Bench separates an 11-domain qualitative split
  from throughput workloads at multiple input lengths and concurrencies, and
  evaluates production engines including vLLM and TensorRT-LLM.
- **Relevance and boundary:** It supports RelaySpec's cross-task acceptance and
  Amdahl analysis and prohibits presenting four independent batch-one ranks as
  serving throughput. The frozen experiment answers the DFlash/EAGLE
  single-request latency question; production-serving claims require a later
  engine integration and SPEED-Bench throughput study, not extrapolation from
  the present measurements.

## DeepSpec and the Qwen3 EAGLE-3 release state

- **Title:** DeepSpec: a full-stack codebase for training and evaluating draft
  models for speculative decoding
- **Status:** official DeepSeek open-source implementation, 2026; software
  artifact, not itself a peer-reviewed paper
- **URL:** https://github.com/deepseek-ai/DeepSpec
- **Pinned implementation:** Git commit
  `005e03b81cec38b7da6399833d609ee89a2587f2`
- **Pinned checkpoints:** `deepseek-ai/eagle3_qwen3_4b_ttt7` revision
  `b0b90fd15d052217c226be5e46d468d8d129e0cd`; Qwen3-8B revision
  `f6485ba8d21e11942958617dbe7e71b467f38f38`; Qwen3-14B revision
  `d7ea05d0b0009badfff0df2dcaedf82cce0f74f8`
- **Checked config locators:** at the pinned commit,
  `config/eagle3/eagle3_qwen3_8b.py` fixes target taps
  `[1,9,17,25,33]`, `ttt_length=7`, learning rate `6e-4`, weight decay zero,
  and gradient clipping one; `config/eagle3/eagle3_qwen3_14b.py` supplies the
  14B tap contract, and `config/dflash/dflash_qwen3_8b.py` independently agrees
  on the optimizer values. These are implementation precedents, not evidence
  that RelaySpec's shorter fit budget is globally optimal.
- **Relevance:** DeepSpec exposes separate Qwen3-4B, 8B, and 14B EAGLE-3
  checkpoints and a common evaluator, making it the implementation base for the
  second-proposer RelaySpec experiment. They are official DeepSpec release
  artifacts, but they must not be misdescribed as official SafeAILab EAGLE
  checkpoints or as peer-reviewed evidence by themselves.
- **Evaluator scope:** the pinned
  `deepspec/eval/eagle3/evaluator.py` performs a length-`ttt` autoregressive
  proposal chain through `generate_decoding_sample`; it does not implement the
  dynamic draft tree used for some EAGLE-3 paper/serving results. RelaySpec's
  EAGLE numbers are therefore compared only with the matched DeepSpec source
  and native-target chain paths, never numerically equated to published tree
  results.

## Sources rechecked for the 5 September manuscript rewrite

- `bansal2021stitching`: Bansal, Nakkiran and Barak, *Revisiting Model Stitching to Compare Neural Representations*, NeurIPS 2021. Peer reviewed. https://papers.nips.cc/paper/2021/hash/01ded4259d101feb739b06c399e9cd9c-Abstract.html . Used to credit learned connections between frozen networks.
- `smith2025stitching`: Smith, Mannering and Marcu, *Functional Alignment Can Mislead: Examining Model Stitching*, ICML 2025, PMLR 267:55972–55998. Peer reviewed. https://proceedings.mlr.press/v267/smith25a.html . Used to distinguish useful behavioral connections from evidence of shared information.
- PARD full paper: https://arxiv.org/html/2504.18583v4 . Confirms target-independent reuse and the separation of drafter adaptation from inference measurements. ICLR 2026 status and public reviews are recorded in the dedicated review study.
- DFlash full paper: https://arxiv.org/html/2602.06036v2 . Used for the drafting-cost/progress explanation and experimental presentation, not as a numerically comparable external baseline.
- EAGLE-3 proceedings: https://proceedings.neurips.cc/paper_files/paper/2025/hash/c7b5a35ea98b62512a869c19ea7b03cb-Abstract-Conference.html . Confirms title, authors and NeurIPS 2025 publication.
- SD² proceedings: https://ojs.aaai.org/index.php/AAAI/article/view/40255 . Confirms AAAI 2026 bibliographic metadata. Frozen and fine-tuned variants are distinguished using the earlier primary-paper audit.
- ICLR 2027 author guidelines: https://iclr.cc/Conferences/2027/AuthorGuidelines . Checked main-text limit, anonymous style, placement of appendices, required AI statement and recommended reproducibility statement. Review-note fetches this turn were blocked by the public site's browser check/HTTP 403. The existing 42-review study is reused and explicitly identified as such.
