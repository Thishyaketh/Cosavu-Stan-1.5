export const defaultPrompt = `What did DeepSeek invented in this Paper

Conditional Memory via Scalable Lookup:
A New Axis of Sparsity for Large Language Models
Xin Cheng1,2, Wangding Zeng2
, Damai Dai2
, Qinyu Chen2
, Bingxuan Wang2
,
Zhenda Xie2
, Kezhao Huang2
, Xingkai Yu2
, Zhewen Hao2
, Yukun Li2
, Han Zhang2
,
Huishuai Zhang1
, Dongyan Zhao1
, Wenfeng Liang2
1Peking University 2DeepSeek-AI
{zhanghuishuai, zhaody}@pku.edu.cn
{chengxin, zengwangding, damai.dai}@deepseek.com
Abstract
While Mixture-of-Experts (MoE) scales capacity via conditional computation, Transformers lack
a native primitive for knowledge lookup, forcing them to inefficiently simulate retrieval through
computation. To address this, we introduce conditional memory as a complementary sparsity
axis, instantiated via Engram, a module that modernizes classic 𝑁-gram embedding for O (1)
lookup. By formulating the Sparsity Allocation problem, we uncover a U-shaped scaling law
that optimizes the trade-off between neural computation (MoE) and static memory (Engram).
Guided by this law, we scale Engram to 27B parameters, achieving superior performance
over a strictly iso-parameter and iso-FLOPs MoE baseline. Most notably, while the memory
module is expected to aid knowledge retrieval (e.g., MMLU +3.4; CMMLU +4.0), we observe
even larger gains in general reasoning (e.g., BBH +5.0; ARC-Challenge +3.7) and code/math
domains (HumanEval +3.0; MATH +2.4). Mechanistic analyses reveal that Engram relieves
the backbone's early layers from static reconstruction, effectively deepening the network for
complex reasoning. Furthermore, by delegating local dependencies to lookups, it frees up
attention capacity for global context, substantially boosting long-context retrieval (e.g., MultiQuery NIAH: 84.2 → 97.0). Finally, Engram establishes infrastructure-aware efficiency: its
deterministic addressing enables runtime prefetching from host memory, incurring negligible
overhead. We envision conditional memory as an indispensable modeling primitive for nextgeneration sparse models. Code available at: https://github.com/deepseek-ai/Engram
1. Introduction
Sparsity is a recurring design principle for intelligent systems, spanning from biological neural
circuits (Lennie, 2003; Olshausen and Field, 1997) to modern Large Language Models (LLMs).
Currently, this principle is primarily realized through Mixture-of-Experts (MoE) (Dai et al., 2024;
Shazeer et al., 2017), which scales capacity via conditional computation. Owing to its ability to
drastically increase model size without proportional increases in compute, MoE has become the
de facto standard for frontier models (Comanici et al., 2025; Guo et al., 2025; Team et al., 2025).
Despite the success of this conditional computation paradigm, the intrinsic heterogeneity
of linguistic signals suggests significant room for structural optimization. Specifically, language
modeling entails two qualitatively different sub-tasks: compositional reasoning and knowlarXiv:2601.07372v1 [cs.CL] 12 Jan 2026
edge retrieval. While the former demands deep, dynamic computation, a substantial portion
of text—such as named entities and formulaic patterns—is local, static, and highly stereotyped (Constant et al., 2017; Erman, 2000). The effectiveness of classical 𝑁-gram models (Brants
et al., 2007; Liu et al., 2024b; Nguyen, 2024) in capturing such local dependencies implies that
these regularities are naturally represented as computationally inexpensive lookups. Since
standard Transformers (Vaswani et al., 2017) lack a native knowledge lookup primitive, current
LLMs are forced to simulate retrieval through computation. For instance, resolving a common
multi-token entity requires consuming multiple early layers of attention and feed-forward networks (Ghandeharioun et al., 2024; Jin et al., 2025) (see Table 3). This process essentially amounts
to an expensive runtime reconstruction of a static lookup table, wasting valuable sequential
depth on trivial operations that could otherwise be allocated to higher-level reasoning.
To align model architecture with this linguistic duality, we advocate for a complementary
axis of sparsity: conditional memory. Whereas conditional computation sparsely activates
parameters to process dynamic logic (Bengio et al., 2013; Shazeer et al., 2017), conditional
memory relies on sparse lookup operations to retrieve static embeddings for fixed knowledge.
As a preliminary exploration of this paradigm, we revisit 𝑁-gram embeddings (Bojanowski et al.,
2017) as a canonical instantiation: local context serves as a key to index a massive embedding
table via constant-time O (1) lookups (Huang et al., 2025a; Pagnoni et al., 2025; Tito Svenstrup
et al., 2017; Yu et al., 2025). Our investigation reveals that, perhaps surprisingly, this static
retrieval mechanism can serve as an ideal complement to modern MoE architecture—but
only if it is properly designed. In this paper, we propose Engram, a conditional memory
module grounded in the classic 𝑁-gram structure but equipped with modern adaptations
such as tokenizer compression, multi-head hashing, contextualized gating, and multi-branch
integration (detailed in Section 2).
To quantify the synergy between these two primitives, we formulate the Sparsity Allocation
problem: given a fixed total parameter budget, how should capacity be distributed between
MoE experts and Engram memory? Our experiments uncover a distinct U-shaped scaling
law, revealing that even simple lookup mechanisms, when treated as a first-class modeling
primitive, act as essential complements to neural computation. Guided by this allocation law, we
scale Engram to a 27B-parameter model. Compared to a strictly iso-parameter and iso-FLOPs
MoE baseline, Engram-27B achieves superior efficiency across diverse domains. Crucially, the
gains are not limited to knowledge-intensive tasks (e.g., MMLU: +3.4; CMMLU: +4.0; MMLUPro: +1.8), where memory capacity is intuitively beneficial; we observe even more significant
improvements in general reasoning (e.g., BBH: +5.0; ARC-Challenge: +3.7; DROP: +3.3) and
code/math domains (e.g., HumanEval: +3.0; MATH: +2.4; GSM8K: +2.2).
Mechanistic analysis via LogitLens (nostalgebraist, 2020) and CKA (Hendrycks et al., 2021a)
reveals the source of these gains: Engram relieves the backbone from reconstructing static
knowledge in early layers, thereby increasing effective depth available for complex reasoning. Furthermore, by delegating local dependencies to lookups, Engram frees up attention
capacity to focus on global context, enabling exceptional performance in long-context scenarios—substantially outperforming baselines on LongPPL (Fang et al.) and RULER (Hsieh et al.)
(e.g., Multi-Query NIAH: 97.0 vs. 84.2; Variable Tracking: 89.0 vs. 77.0).
Finally, we establish infrastructure-aware efficiency as a first-class principle. Unlike MoE's
dynamic routing, Engram employs deterministic IDs to enable runtime prefetching, overlapping
communication with computation. Empirical results show that offloading a 100B-parameter
table to host memory incurs negligible overhead (< 3%). This demonstrates that Engram
effectively bypasses GPU memory constraints, facilitating aggressive parameter expansion.
2
Vocab Embedding
Only Alexander the Great could tame the horse Bucephalus
Transformer Block
.
2-Gram Embedding 3-Gram Embedding
h
Concat
Linear
h Hash Hash
Scaled Dot Product
Input Hidden
Linear
Engram
Attention
MoE
Conv
the Great Alexander the Great
Figure 1 | The Engram Architecture. The module augments the backbone by retrieving static 𝑁-
gram memory and fusing it with dynamic hidden states via context-aware gating. This module
is applied only to specific layers to decouple memory from compute, leaving the standard input
embedding and un-embedding module intact.
2. Architecture
2.1. Overview
As shown in Figure 1, Engram is a conditional memory module designed to augment the Transformer backbone by structurally separating static pattern storage from dynamic computation.
Formally, given an input sequence 𝑋 = (𝑥1, . . . , 𝑥𝑇 ) and hidden states H(ℓ) ∈ R𝑇×𝑑 at layer ℓ,
the module processes each position 𝑡 in two functional phases: retrieval and fusion. First, as
detailed in Section 2.2, we extract and compress suffix 𝑁-grams to deterministically retrieve
static embedding vectors via hashing. Subsequently, in Section 2.3, these retrieved embeddings
are dynamically modulated by the current hidden state and refined via a lightweight convolution. Finally, we discuss the integration with multi-branch architectures in Section 2.4 and the
system-level design in Section 2.5.
2.2. Sparse Retrieval via Hashed 𝑁-grams
The first phase maps local contexts to static memory entries, involving tokenizer compression
and retrieving embeddings via deterministic hashing.
Tokenizer Compression While 𝑁-gram models typically operate directly on tokenizer outputs,
standard subword tokenizers prioritize lossless reconstruction, often assigning disjoint IDs to
semantically equivalent terms (e.g., Apple vs. ␣apple) (Kudo and Richardson, 2018; Li et al.,
2023b). To maximize semantic density, we implement a vocabulary projection layer. Specifically,
we pre-compute a surjective function P : 𝑉 → 𝑉
′
that collapses raw token IDs into canonical
3
identifiers based on normalized textual equivalence (using NFKC (Whistler, 2025), lowercasing,
etc.). In practice, this process achieves a 23% reduction in the effective vocabulary size for a
128k tokenizer (see Appendix C). Formally, for a token at position 𝑡, we map its raw ID 𝑥𝑡 to a
canonical ID 𝑥
′
𝑡 = P (𝑥𝑡) to form the suffix 𝑁-gram 𝑔𝑡,𝑛 = (𝑥
′
𝑡−𝑛+1
, . . . , 𝑥
′
𝑡
).
Multi-Head Hashing. Directly parameterizing the combinatorial space of all possible 𝑁-grams
is intractable. Following Tito Svenstrup et al. (2017), we adopt a hashing-based approach. To
mitigate collisions, we employ 𝐾 distinct hash heads for each 𝑁-gram order 𝑛. Each head 𝑘 maps
the compressed context to an index within an embedding table E𝑛,𝑘 (of prime size 𝑀𝑛,𝑘) via a
deterministic function 𝜑𝑛,𝑘.

2.3. Context-aware Gating
The retrieved embeddings serve as context-independent priors. Being static, however, they
inherently lack contextual adaptability and may suffer from noise due to hash collisions or
polysemy (Haber and Poesio, 2024). To enhance expressivity and resolve this ambiguity, we
employ a context-aware gating mechanism inspired by Attention (Bahdanau et al., 2015; Vaswani
et al., 2017). The current hidden state—which has aggregated global context via preceding
attention layers—is used as a dynamic Query, while the retrieved memory serves as the source
for both Key and Value projections. We apply RMSNorm to the Query and Key before computing
a scalar gate via sigmoid. The gated output is then refined via a short, depthwise causal
convolution with SiLU activation, and integrated into the backbone via a residual connection
followed by the standard Attention and MoE. Engram is not applied to every layer; its specific
placement is governed by system-level latency constraints.

2.4. Integration with Multi-branch Architecture
We adopt an advanced multi-branch architecture (Manifold-Constrained Hyper-Connections,
M=4) as the default backbone. A single sparse embedding table and a Value projection matrix
W_V are shared across all M branches, whereas M distinct Key projection matrices are employed
to enable branch-specific gating behaviors. This design allows linear projections to be fused
into a single dense FP8 matmul, maximizing compute utilization of modern GPUs.

2.5. System Efficiency: Decoupling Compute and Memory
Unlike MoE, which relies on runtime hidden states for dynamic routing, Engram's retrieval
indices depend solely on the input token sequence. During training, embedding tables are
sharded across GPUs with All-to-All communication. During inference, a prefetch-and-overlap
strategy asynchronously retrieves embeddings from host memory via PCIe, masking communication latency with the computation of preceding Transformer blocks. Natural language
N-grams follow a Zipfian distribution, motivating a Multi-Level Cache Hierarchy where
frequently accessed embeddings reside in HBM/DRAM and the long tail in NVMe SSD.

3. Scaling Laws and Sparsity Allocation
We define the allocation ratio ρ ∈ [0,1] as the fraction of the inactive-parameter budget
assigned to MoE expert capacity, with the remainder allocated to Engram embedding slots.
Experiments at two compute regimes (2e20 and 6e20 FLOPs) reveal a consistent U-shaped
relationship between validation loss and ρ. The pure MoE baseline (ρ=100%) is suboptimal:
reallocating roughly 20–25% of the sparse parameter budget to Engram yields the best
performance, with the optimum stable around ρ ≈ 75–80% across regimes. Under aggressive
memory scaling (Infinite Memory Regime), validation loss exhibits a log-linear power law
with respect to the number of embedding slots — Engram unlocks much larger scaling
potential than OverEncoding-style averaging.

4. Large Scale Pre-training
We train four models on 262B tokens: Dense-4B (4.1B), MoE-27B (26.7B, 72 routed + 2 shared
experts, top-6), Engram-27B (26.7B, 55 routed + 2 shared experts + 5.7B Engram memory),
Engram-40B (39.5B, +18.5B Engram memory). All match 3.8B activated parameters. Engram-27B
consistently improves over the iso-parameter and iso-FLOPs MoE-27B baseline. Gains include:
MMLU +3.0, MMLU-Pro +1.8, CMMLU +4.0, BBH +5.0, ARC-Challenge +3.7, DROP +3.3,
HumanEval +3.0, MBPP +1.6, GSM8K +2.2, MATH +2.4. Engram-40B further reduces
pre-training loss across most benchmarks.

5. Long Context Training
Following pre-training, we apply YaRN for 32k-token context extension over 5,000 steps (30B
tokens). Under the Iso-Loss setting (Engram-27B at 46k steps vs. MoE-27B at 50k steps,
matched pre-training loss), Engram demonstrates significant gains on RULER: Multi-Query
NIAH 97.0 vs. 84.2; Variable Tracking 87.2 vs. 77.0; FWE 98.6 vs. 73.0. Under the Iso-FLOPs
setting (Engram-27B at 50k), gains widen further across all metrics. Even at 82% compute
(Engram-27B at 41k), the model matches MoE-27B (50k) on LongPPL and surpasses it on RULER.

6. Analysis
6.1 Effective Depth. LogitLens shows Engram variants exhibit systematically smaller layer-wise
KL divergence to the final output, indicating faster prediction convergence. CKA analysis reveals
a distinct off-diagonal upward shift — e.g., representations at layer 5 of Engram-27B align with
approximately layer 12 of MoE-27B — confirming that Engram is functionally equivalent to
increasing effective model depth.

6.2 Structural Ablation. Sweeping a single 1.6B Engram module across 12 layers shows Layer 2
is optimal under a single-injection constraint, with Layer 1 and deeper layers degrading. Splitting
the budget across Layers 2 and 6 outperforms any single injection. Removing multi-branch
fusion, context-aware gating, or tokenizer compression causes the largest regressions.

6.3 Sensitivity. Suppressing Engram at inference causes factual-knowledge benchmarks
(TriviaQA, PopQA) to collapse to 29–44% of baseline, confirming Engram acts as the primary
repository for parametric knowledge. Reading comprehension tasks (C3, RACE) retain 81–93%,
showing reliance on backbone attention rather than Engram.

6.4 System Efficiency. Offloading a 100B-parameter Engram layer entirely to host DRAM
on an NVIDIA H800 incurs only a 1.9–2.8% throughput penalty (4B-Dense: 9,031 → 8,858
tok/s; 8B-Dense: 6,315 → 6,140 tok/s). Hierarchical caching (Zipfian locality) would reduce
overhead further.

6.5 Gating Visualization. The gating scalar α activates strongly on multi-token named entities
("Alexander the Great", "the Milky Way", "Princess of Wales") and formulaic phrases ("By the
way"). The pattern generalizes to Chinese, with strong activations on idioms and historical
entities (Four Great Inventions, Zhang Zhongjing).

7. Related Work
N-gram modeling and embedding scaling: SuperBPE merges multi-word expressions; SCONE
uses an auxiliary encoding model; OverEncoding and Byte Latent Transformer (BLT) adopt
hash N-gram embeddings. Engram differs by (a) treating conditional memory as a first-class
primitive evaluated under strict iso-parameter and iso-FLOPs constraints, and (b) algorithm-
system co-design that injects memory into deeper layers to enable communication-computation
overlap. Other prior work: MoE (Shazeer 2017, GShard, Switch Transformer, GLaM,
DeepSeekMoE); Memory networks (PKM, PEER, UltraMem; REALM, RETRO, PlugLM);
Knowledge storage mechanisms (FFNs as key-value memories, ROME, MEMIT).

8. Conclusion
We introduce conditional memory as a complementary sparsity axis to conditional computation
(MoE), instantiated via Engram — modernized N-gram embeddings enabling scalable, constant-
time O(1) lookups for static patterns. The Sparsity Allocation problem reveals a U-shaped
scaling law where hybrid MoE+Engram strictly outperforms pure MoE. Mechanistic analysis
shows Engram effectively deepens the network by relieving early layers from static
reconstruction, freeing attention capacity for global context and complex reasoning, yielding
substantial long-context gains. Deterministic addressing enables offloading massive parameter
tables to host memory with negligible inference overhead. We envision conditional memory as
an indispensable modeling primitive for next-generation sparse models.`;
