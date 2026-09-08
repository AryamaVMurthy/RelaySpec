# Source log

Checked 2026-09-08. Targeted search, not an exhaustive systematic review. Bibliographic keys refer to references.bib; arxiv-metadata.json preserves metadata fetched directly from primary arXiv pages. Public preprints are relevant prior art even when venue status is unverified. Published speed claims were not treated as matched local comparisons.

## arxiv221117192: Fast Inference from Transformers via Speculative Decoding

- Authors: Leviathan, Yaniv, Kalman, Matan, Matias, Yossi
- URL: https://arxiv.org/abs/2211.17192
- Status: ICML 2023 peer-reviewed paper; arXiv version metadata retained
- Reading scope: Primary abstract and proceedings
- Relevance: Exact speculative decoding distribution guarantee; proceedings checked.

## arxiv231008461: DistillSpec: Improving Speculative Decoding via Knowledge Distillation

- Authors: Zhou, Yongchao, Lyu, Kaifeng, Rawat, Ankit Singh, Menon, Aditya Krishna, Rostamizadeh, Afshin, Kumar, Sanjiv, Kagy, Jean-François, Agarwal, Rishabh
- URL: https://arxiv.org/abs/2310.08461
- Status: ICLR 2024 peer-reviewed paper; arXiv version metadata retained
- Reading scope: Primary abstract and conference PDF
- Relevance: Distillation to improve draft/target agreement predates this study.

## arxiv260412989: Accelerating Speculative Decoding with Block Diffusion Draft Trees

- Authors: Ringel, Liran, Romano, Yaniv
- URL: https://arxiv.org/abs/2604.12989
- Status: Primary arXiv preprint version; venue status not independently verified
- Reading scope: Full-text method inspected
- Relevance: Direct source of the adapted local heap-builder baseline.

## arxiv260601813: Cost-Aware Diffusion Draft Trees for Speculative Decoding

- Authors: Zhang, Shuai, Qiu, Huachuan, He, Hongliang, Dai, Yong
- URL: https://arxiv.org/abs/2606.01813
- Status: Primary arXiv preprint version; venue status not independently verified
- Reading scope: Full-text method and cost model inspected
- Relevance: Cost-aware diffusion tree budgeting; missing direct control.

## arxiv260830135: Verification-Aware Training for Speculative Decoding

- Authors: Gu, Geonmo, Heo, Byeongho, Jun, HeeJae, Kang, Yoohoon, Lee, Sangmin, Yun, Sangdoo, Han, Dongyoon
- URL: https://arxiv.org/abs/2608.30135
- Status: Primary arXiv preprint version; venue status not independently verified
- Reading scope: Full-text objective inspected
- Relevance: Sequential-verification supervision and adaptive training weights.

## arxiv260618394: JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting

- Authors: Hu, Lanxiang, Feng, Zhaoxiang, Wu, Yulun, Yuan, Haoran, Zhao, Yujie, Qian, Yu-Yang, Wang, Bojun, Zhao, Peng, Jiang, Daxin, Zhu, Yibo, Rosing, Tajana, Zhang, Hao
- URL: https://arxiv.org/abs/2606.18394
- Status: Primary arXiv preprint version; venue status not independently verified
- Reading scope: Full-text architecture and official source inspected
- Relevance: Causal parallel tree drafting; algorithm/runtime distinction matters.

## arxiv260706763: Trees from Marginals: Autoregressive drafting with factorized priors

- Authors: Oda, Yuma, Mathieu, Ryan, Knyazhitskiy, Roman, Chakhvadze, Artur
- URL: https://arxiv.org/abs/2607.06763
- Status: Primary arXiv preprint version; venue status not independently verified
- Reading scope: Primary abstract
- Relevance: Lightweight autoregressive adapter over factorized candidate marginals.

## arxiv260802123: From Chains to Trees: Parent-Conditioned Drafting for Semi-Autoregressive Speculative Decoding

- Authors: Li, Zixian, Li, Tong, Xie, Chi, Song, Xiaohui, Lu, Haonan
- URL: https://arxiv.org/abs/2608.02123
- Status: Primary arXiv preprint version; venue status not independently verified
- Reading scope: Primary abstract
- Relevance: Parent-conditioned branching from an existing Markov head.

## arxiv260813524: DARTree: Speculative Diffusion Decoding with Autoregressive Draft Trees

- Authors: Li, Tianyi, Luo, Yaxin, Shang, Xinyi, Shen, Zhiqiang
- URL: https://arxiv.org/abs/2608.13524
- Status: Primary arXiv preprint version; venue status not independently verified
- Reading scope: Primary abstract
- Relevance: Extends pretrained chain correction to tree drafting.

## arxiv260119278: DART: Diffusion-Inspired Speculative Decoding for Fast LLM Inference

- Authors: Liu, Fuliang, Li, Xue, Zhao, Ketai, Gao, Yinxi, Zhou, Ziyan, Zhang, Zhonghui, Wang, Zhibin, Dou, Wanchun, Zhong, Sheng, Tian, Chen
- URL: https://arxiv.org/abs/2601.19278
- Status: Primary arXiv preprint version; venue status not independently verified
- Reading scope: Primary abstract; earlier official source inspection
- Relevance: Continuity-aware trees overlap our history heuristic.

## arxiv260117768: LLM-42: Enabling Determinism in LLM Inference with Verified Speculation

- Authors: Gond, Raja, Kamath, Aditya K, Ramjee, Ramachandran, Panwar, Ashish
- URL: https://arxiv.org/abs/2601.17768
- Status: ArXiv version checked; author repository reports SOSP 2026 acceptance, conference listing not verified
- Reading scope: Primary full-text description and official microsoft/llm-42 repository
- Relevance: Verified fixed-shape replay for deterministic inference.

## arxiv260600487: TAPS: Target-Aware Prefix Tree Selection for Diffusion-Drafted Speculative Decoding

- Authors: Wang, Zhuoyu, Huang, Junnan, Chen, Xinyu
- URL: https://arxiv.org/abs/2606.00487
- Status: Primary arXiv preprint version; venue status not independently verified
- Reading scope: Primary abstract
- Relevance: Target-aware prefix-conditioned tree selection.

## arxiv260705147: DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation

- Authors: Cheng, Xin, Yu, Xingkai, Shao, Chenze, Li, Jiashi, Xiong, Yunfan, Qian, Yi, Zhu, Jiaqi, Ma, Shirong, Zhang, Xiaokang, Ye, Jiasheng, Chen, Qinyu, Deng, Chengqi, Yu, Jiping, Dai, Damai, Zhang, Zhengyan, Wei, Yixuan, Tan, Yixuan, Yang, Wenkai, Xu, Runxin, Wu, Yu, Xu, Zhean, Wang, Xuanyu, Chen, Muyang, Tian, Rui, Bi, Xiao, Hao, Zhewen, Chen, Shaoyuan, Cao, Huanqi, Zhang, Wentao, Xu, Anyi, Zhang, Huishuai, Zhao, Dongyan, Liang, Wenfeng
- URL: https://arxiv.org/abs/2607.05147
- Status: Primary arXiv preprint version; venue status not independently verified
- Reading scope: Primary abstract
- Relevance: Semi-autoregressive drafting and confidence/load-aware scheduling.

## arxiv260820375: GRAFT: Adaptive DLM-Based Draft Tree Construction with Target-Distilled Edge Scoring

- Authors: Ye, Xuming, Ma, Zeming, Yu, Runjie, Liu, Yuan, Li, Tianle, Bai, Shuhan, Zhou, Jian, Wu, Fei
- URL: https://arxiv.org/abs/2608.20375
- Status: Primary arXiv preprint version; venue status not independently verified
- Reading scope: Primary title/metadata and discovery abstract only
- Relevance: Additional target-distilled tree-scoring work identified for follow-up; no detailed evaluation claim here.

## arxiv260717283: Lossless but Not Free: An Empirical Anatomy of Speculative Decoding on Consumer Hardware

- Authors: Chordiya, Param
- URL: https://arxiv.org/abs/2607.17283
- Status: Primary arXiv preprint version; venue status not independently verified
- Reading scope: Primary abstract
- Relevance: Existing empirical analysis includes successful and slower deployment configurations.

## Additional primary sources

- `iclr2027guide`: https://iclr.cc/Conferences/2027/ReviewerGuidelines — official venue criteria; significance does not require SOTA.
- `pytorch29numerical`: https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html — official runtime documentation on shape-dependent floating-point results.
- `salf_talf`: https://openreview.net/pdf?id=3V559xWIWc — public anonymous manuscript headed “Under review as a conference paper at ICLR 2026”; current decision and authors not verified. Abstract explicitly identifies a tree-aware loss. Do not label it an accepted conference paper.
- LLM-42 acceptance statement: https://github.com/microsoft/llm-42 — author repository, not independently verified proceedings metadata.

## Search queries and scope

Queries covered exact speculative-decoding distributions; DDTree/JetSpec and direct references; numerical precision and determinism; cost-aware diffusion draft trees; acceptance-aware training; draft compression; tree-aware loss; prefix-survival objectives; target-aware prefix selection; ICLR 2027 reviewer criteria. Primary arXiv full texts were inspected for DDTree, CaDDTree, VAT, JetSpec and LLM-42. Other entries are explicitly marked abstract-only or metadata-only. Secondary discovery snippets were used to find primary pages, not as technical evidence. No systematic citation-completeness claim is made.

Local evidence: ../DECISION.md, ../confirmation-summary.json, ../quality-summary.json, ../run-29337/lane0/precision-result.json, and ../../ddtree_baseline.py / ../../radical_decode.py.
