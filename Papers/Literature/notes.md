# Literature Notes — Bachelor Thesis
**Thesis:** Comparison of a Classical RAG Pipeline and an Agent-Based Approach for Intelligent Knowledge Retrieval Across Enterprise Documentation Sources

This file aggregates analysis reports from all literature papers. Each entry follows the same structure: core argument, key points keyed to thesis chapters, quotable passages, and critical notes on limitations.

---

## Part 1 — RAG Foundations

### Lewis, P., Perez, E., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
**Citation key:** lewis2020rag  
**arXiv:** 2005.11401  
**Thesis relevance:** Ch 2.2 (core RAG pipeline definition), Ch 3.2 (RAG architecture design), Ch 7.4 (connection to prior work)  
**Local file:** `Papers/Literature/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.md`

#### Core Argument
Lewis et al. introduce Retrieval-Augmented Generation (RAG) as a general-purpose framework that combines a parametric memory component (a pre-trained seq2seq language model, specifically BART) with a non-parametric memory component (a dense vector index over Wikipedia, retrieved via DPR) to ground text generation in external knowledge. The central claim is that this hybrid architecture reduces factual hallucinations and enables knowledge updating without retraining, while remaining end-to-end trainable. RAG achieves state-of-the-art results on open-domain QA and other knowledge-intensive tasks at the time of publication.

#### Key Points for the Thesis
- **Architecture decomposition:** RAG consists of two components: a retriever `p_η(z|x)` parameterised by DPR (Dense Passage Retrieval), which maps an input query `x` to a distribution over text passages `z` from a fixed corpus; and a generator `p_θ(y_i|x, z, y_{1:i-1})` parameterised by BART-large, which conditions on both the query and retrieved passages to produce output tokens. This retriever-generator decomposition is the direct ancestor of all classical RAG pipelines the thesis compares against.
- **Two marginalization strategies:** RAG-Sequence marginalises over documents at the sequence level — the same document is used for the entire generated answer, and the final probability is a sum over top-K document hypotheses. RAG-Token marginalises at each token step, allowing different documents to inform different parts of the output. RAG-Token is more flexible but more expensive; this trade-off is directly relevant to Ch 6 latency discussion.
- **Dense retrieval over sparse:** The use of DPR instead of BM25 is an explicit design choice. Lewis et al. report that dense retrieval provides significantly better passage coverage for knowledge-intensive tasks than TF-IDF/BM25 baselines, establishing the empirical case for dense embedding-based retrieval that Ch 2.2 should anchor.
- **Non-parametric knowledge update:** A key claimed advantage is that the document index can be updated without retraining the generator — relevant to Ch 7 discussion of maintainability: updating an index is simpler than retraining, but index quality and freshness still require operational investment.
- **Results:** RAG sets new state-of-the-art on Natural Questions (44.5 EM), WebQuestions (45.5 EM), CuratedTrec (56.8 EM), and TriviaQA (68.0 EM) in the open-domain setting. Results are in clean, single-source settings — not multi-source heterogeneous enterprise contexts.
- **Training:** The retriever and generator are jointly fine-tuned end-to-end using FAISS-indexed MIPS, with the document encoder kept frozen during training.
- **Failure mode acknowledged:** The paper notes that RAG can still generate text not supported by retrieved passages, because the generator is not explicitly trained to stay faithful to retrieved content. This is the gap that SELF-RAG later addresses.

#### Quotable Passages
- Abstract: "We build RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever."
- Section 4: "Unlike T5 or GPT, RAG's non-parametric memory can be directly revised and expanded, and our knowledge is interpretable and can be inspected and modified."
- Section 2: "For the RAG-Sequence model, to generate each target token, the model attends to the same document. For the RAG-Token model, the model marginalizes over the top-K documents independently for each target token."
- Discussion: "RAG combines the benefits of both worlds — the flexibility and generalization of a parametric model with the interpretability and updateability of non-parametric retrieval."

#### Critical Notes
- Evaluates exclusively on clean, homogeneous corpora (Wikipedia) — no multi-source retrieval, no source-type routing, no structural heterogeneity between document types. Fundamental gap relative to the enterprise thesis setting with GitHub Issues, documentation, and PDFs.
- Single retrieval step per query. No mechanism to recognise when retrieval is unnecessary, irrelevant, or insufficient. This "indiscriminate retrieval" is the specific limitation SELF-RAG and FLARE subsequently critique.
- The end-to-end training requirement means RAG as originally formulated is not plug-and-play. Modern pipeline RAG (which this thesis implements) departs from this by using off-the-shelf LLMs with retrieved context prepended in the prompt.
- The paper does not address chunking, reranking, context compression, or query reformulation — all central to Ch 3.2 design decisions. Gao et al. 2023 addresses these gaps.
- No mechanism for multi-hop retrieval or cross-document synthesis.

---

### Gao, Y., et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey
**Citation key:** gao2023ragsurvey
**arXiv:** 2312.10997 (submitted December 2023, last revised March 2024)
**Full citation:** Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M., & Wang, H. (2024). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv preprint arXiv:2312.10997*.
**Thesis relevance:** Ch 2.2 (RAG taxonomy and pipeline components), Ch 2.4 (evaluation dimensions), Ch 3.2 (chunking and embedding decisions), Ch 7.4 (framing of Modular RAG)  
**Local file:** `Papers/Literature/Retrieval-Augmented Generation for Large Language Models A Survey.md`

#### Core Argument
Gao et al. provide a comprehensive taxonomy of RAG techniques up to late 2023, organising the field into three paradigms (Naive, Advanced, Modular RAG), surveying the technical components of each pipeline stage, and defining a rigorous four-dimension evaluation framework. The paper argues that while Naive RAG has fundamental limitations in retrieval precision and generation faithfulness, Advanced and Modular RAG variants address these through improved chunking, retrieval, reranking, and self-reflection mechanisms, and that the field is converging toward systems with increasing agent-like properties. The survey frames this progression as a maturation toward systems capable of autonomous source selection, query reformulation, and iterative retrieval — the very capabilities this thesis attributes to agent-based systems.

#### The RAG Taxonomy (Critical for Ch 2.2)

**Naive RAG** is the baseline paradigm: documents are chunked, embedded, and stored in a vector database; at query time the query is embedded, top-K chunks are retrieved by cosine similarity, and the retrieved chunks plus query are passed to an LLM for generation. Its limitations are well-defined: low retrieval precision (irrelevant chunks are included), recall gaps (relevant content may not surface), no query reformulation, and inability to handle multi-step reasoning. Single retrieval is performed once per query with no feedback loop.

**Advanced RAG** preserves the index-retrieve-generate structure but adds targeted mechanisms before and after retrieval. Pre-retrieval improvements include query rewriting, query expansion, and indexing optimizations such as sliding window chunking, fine-grained segmentation, and metadata attachment. Post-retrieval improvements include reranking (to promote the most relevant chunks to the top of the context) and context compression (to reduce noise and prevent the "lost in the middle" problem where LLMs fail to use information buried in long contexts). Advanced RAG remains a pipeline — it does not decide *whether* to retrieve, only *how* to retrieve better.

**Modular RAG** deconstructs the pipeline into interchangeable components and introduces new modules not present in earlier paradigms. Key additions include: a *search module* that can query diverse source types (databases, knowledge graphs, APIs, search engines); a *memory module* that uses the LLM's own outputs to iteratively refine retrieval; a *routing component* that selects the appropriate data source or retrieval strategy based on query characteristics; and a *task adapter* for downstream task customization. New operational patterns emerge: Rewrite-Retrieve-Read (query refinement before retrieval), Generate-Read (LLM generates context rather than retrieving it), hybrid retrieval (combining sparse BM25 with dense vector search), and — most critically — *adaptive retrieval*, where the system autonomously decides whether retrieval is needed at all. Adaptive retrieval is the point at which Modular RAG begins to overlap with agent-based behavior, which is directly relevant to the Ch 7.4 discussion.

**Positioning the classical pipeline:** The thesis's classical RAG system sits squarely in the Advanced RAG paradigm — it retrieves, reranks, and compresses, but does not route between sources or decide when to retrieve. This is the cleanest framing of the distinction against the agent system.

#### Four Evaluation Dimensions (Critical for Ch 2.4)

These four dimensions assess RAG-specific capabilities beyond standard QA metrics. They are distinct from RAGAS quality scores (context relevance, faithfulness, answer relevance) and specifically test robustness behaviors.

**Noise Robustness:** The system's ability to handle retrieved documents that are topically related to the query but contain no useful answer — "noisy" documents. A robust system neither discards useful signal because some documents are noisy, nor is it misled into producing an incorrect answer by those documents. This is particularly relevant when retrieval surfaces many near-miss documents.

**Negative Rejection:** The system's ability to withhold an answer when retrieved documents genuinely do not contain the information needed to answer the question. Rather than hallucinating a plausible-sounding response from weak context, a system with strong negative rejection will explicitly acknowledge that no answer can be given. This tests calibration — knowing what you don't know.

**Information Integration:** The system's ability to synthesise information spread across multiple retrieved documents to answer a question that no single document can resolve alone. This directly maps onto multi-hop queries in the evaluation set and is one of the key dimensions where agent-based systems are hypothesised to outperform static RAG (because agents can iteratively retrieve and reason, while RAG performs a single retrieval pass).

**Counterfactual Robustness:** The system's ability to identify and discount factually incorrect information within retrieved documents, even when that incorrect content appears in the retrieved context. This tests whether the system relies blindly on retrieval or can apply prior knowledge to filter out bad context. Particularly relevant when knowledge bases contain outdated or inconsistent documents — a realistic scenario in enterprise environments.

**Usage note for Ch 2.4:** Present these four dimensions as the RAG-specific evaluation layer *beneath* standard NLP metrics, and explain why the thesis evaluation adds latency, cost, and complexity as additional dimensions that the Gao et al. framework does not address.

#### Key Points on Chunking and Embedding (For Ch 3.2)

**Chunking trade-offs (directly from the survey):**
- Fixed-token chunking at 100–512 tokens is the most common baseline. Larger chunks preserve more context but introduce noise and increase LLM processing cost. Smaller chunks are cleaner but may lack sufficient context for the LLM to generate a coherent answer.
- Sliding window chunking overlaps adjacent chunks, reducing the risk that relevant content is split across a boundary — important when sentence-level retrieval units are used.
- The Small2Big approach uses small units (sentences) as retrieval units for precision, but expands the context window delivered to the LLM to include surrounding sentences. This is the best available balance of retrieval precision and generation quality according to the survey.
- Metadata enrichment (source URL, document type, timestamp, category) enables filtered retrieval and is particularly important for multi-source pipelines — directly relevant to the heterogeneous source design.

**Embedding model guidance:**
- Hybrid retrieval (sparse BM25 + dense vector) consistently outperforms either method alone, especially on queries containing rare entities, technical jargon, or short exact-match terms — all common in enterprise documentation.
- BGE (BAAI General Embedding) is specifically mentioned as a strong open-source option. It appears on the MTEB leaderboard and is cost-free, making it the best choice for the thesis given the zero-budget constraint.
- Fine-tuning embedding models on domain-specific data improves retrieval quality, but requires labeled retrieval pairs. Off-the-shelf BGE is the practical choice; fine-tuning can be noted as a future work direction.

**Recommendation for Ch 3.2:** Justify the chunking choice by citing the Small2Big vs. fixed-token trade-off from Gao et al., and justify the embedding model choice (BGE or equivalent) by citing the MTEB leaderboard recommendation and hybrid retrieval advantage.

#### Quotable Passages

Verify all quotes against the PDF at https://arxiv.org/pdf/2312.10997 before including in the thesis.

1. **On RAG's core motivation (Ch 2.2 opening):**
   > "RAG enhances LLMs by incorporating knowledge from external databases, thereby improving the accuracy and credibility of the generation, particularly for knowledge-intensive tasks, and allowing for continuous updates to knowledge." — Abstract

2. **On Naive RAG's limitations (Ch 2.2, positioning Advanced RAG):**
   > "Naive RAG faces several challenges in both retrieval and generation phases. In terms of retrieval, the accuracy of retrieval is problematic... In terms of generation, the model may encounter the issue of hallucination, where it generates content that contradicts the retrieved context."

3. **On Modular RAG's adaptive capability (Ch 7.4, framing the agent comparison):**
   > "Modular RAG advances beyond the previous two paradigms by offering greater adaptability and versatility... adaptive retrieval enables autonomous determination of when retrieval is necessary." — This is the point at which Modular RAG begins to converge with agent-based behavior, making it the natural bridging concept in the discussion chapter.

4. **On chunking granularity (Ch 3.2):**
   > "The granularity of chunks influences the accuracy of retrieval... smaller chunks can lead to more precise retrieval but may lack sufficient context, while larger chunks offer more context but may introduce more noise."

#### Critical Notes

**What the survey identifies as RAG's persistent limitations — these directly motivate the agent system:**

1. **Single-pass retrieval ceiling:** Even Advanced RAG retrieves once (or a fixed number of times) per query. For complex multi-hop questions, a single retrieval cannot surface all necessary information, and there is no mechanism for the system to recognize this and retrieve again.

2. **Static source routing:** RAG pipelines do not decide *which* source to query — they query one vector index. For heterogeneous enterprise environments with structurally distinct source types (tickets vs. wikis vs. PDFs), a single index conflates source-specific retrieval semantics. The agent system addresses this directly by routing to source-specific tools.

3. **No self-correction mechanism:** A Naive or Advanced RAG system that retrieves poor context will generate a poor answer with no awareness of the failure. Modular RAG begins to address this with memory modules, but it remains a limitation relative to agent systems that can observe tool outputs and decide to retry or try a different source.

4. **Chunking as a hard constraint on retrieval quality:** Every chunking strategy involves a fundamental trade-off (context vs. noise), and the optimal chunk size is document-type-dependent. This is particularly problematic for heterogeneous sources, where a single chunking configuration cannot be optimal for all source types simultaneously.

5. **Evaluation gap:** The survey covers 26 tasks and ~50 datasets but does not address latency, token cost, or implementation complexity as first-class evaluation dimensions. This is a genuine contribution gap the thesis fills.

**How to use this in Ch 7.4:** Situate findings within Gao et al.'s Modular RAG framing by arguing that the agent system implements what Modular RAG describes as "adaptive retrieval" and "routing" — but as an external orchestration layer rather than an internal pipeline component. This positions the agent as a natural extension of the RAG evolution they describe, rather than a fundamentally different paradigm.

---

### Asai, A., et al. (2023). SELF-RAG: Learning to Retrieve, Generate, and Critique Through Self-Reflection
**Citation key:** asai2023selfrag  
**arXiv:** 2310.11511  
**Thesis relevance:** Ch 2.2 (critique of static RAG pipelines), Ch 2.3 (transition argument to agent-based approaches), Ch 3.3 (agent design: adaptive retrieval and self-critique), Ch 7.4 (connection to prior work)

#### Core Argument
SELF-RAG trains a single language model to adaptively decide when retrieval is needed, evaluate the relevance of retrieved passages, assess whether its own generated output is supported by retrieved evidence, and score the overall utility of the response — all through special "reflection tokens" emitted as part of the generation process. Unlike standard RAG which retrieves indiscriminately on every query, SELF-RAG retrieves on demand, and unlike standard RAG which does not verify faithfulness, SELF-RAG explicitly self-critiques. Across six diverse tasks, SELF-RAG (7B and 13B parameters) outperforms both retrieval-augmented ChatGPT and conventional RAG baselines.

#### Key Points for the Thesis
- **The core problem with static RAG (most quotable):** "Indiscriminately retrieving and incorporating a fixed number of retrieved passages, regardless of whether retrieval is necessary, or passages are relevant, diminishes LM versatility or can lead to unhelpful response generation." Use in Ch 2.2 as motivation for adaptive approaches, and in Ch 7.2 when discussing when static RAG fails.
- **Four reflection token types:**
  1. **Retrieve** (`{yes, no, continue}`): Decide whether to invoke retrieval.
  2. **ISREL** (`{relevant, irrelevant}`): Evaluate whether retrieved passage is useful for the query.
  3. **ISSUP** (`{fully supported, partially supported, no support}`): Evaluate whether generated content is entailed by retrieved passage.
  4. **ISUSE** (`{1, 2, 3, 4, 5}`): Five-point overall utility score.
  These four token types map onto the thesis's evaluation dimensions: retrieval necessity judgement, retrieval quality, faithfulness, and answer quality.
- **Key results:** SELF-RAG 7B outperforms retrieval-augmented ChatGPT on PopQA (54.9 vs 50.8), PubHealth (72.4 vs 54.7), biography FactScore (81.2 vs 71.8). On PubHealth, gains +22 accuracy over standard RAG baselines.
- **Ablation:** Removing the critic model (equivalent to conventional RAG with always-on retrieval) drops PopQA from 54.9 to 42.6, confirming self-reflection mechanism is the source of gains.
- **SELF-RAG can be understood as agent-like:** It internalises retrieval decision-making and self-critique into the generation model itself, rather than externalising them as separate agent tools and reasoning steps. Use in Ch 2.3 as the bridge concept — the logic an agent performs externally (decide-retrieve-critique) can be internalised.

#### Quotable Passages
- Abstract: "Despite their remarkable capabilities, large language models (LLMs) often produce responses containing factual inaccuracies due to their sole reliance on the parametric knowledge they encapsulate."
- Section 1: "Indiscriminately retrieving and incorporating a fixed number of retrieved passages, regardless of whether retrieval is necessary, or passages are relevant, diminishes LM versatility or can lead to unhelpful response generation."
- Section 1: "We introduce Self-Reflective Retrieval-Augmented Generation (SELF-RAG) that enhances an LM's quality and factuality through retrieval and self-reflection... trains a single arbitrary LM that adaptively retrieves passages on-demand."
- Table 1: "ISSUP: All of the verification-worthy statement in y is supported by d." — the formal definition of faithfulness as an explicit generation objective.
- Section 5.2: "Using the top passages regardless of their relevance (Retrieve top 1) as in conventional RAG approaches causes a large drop in PopQA and ASQA."

#### Critical Notes
- Requires fine-tuning a dedicated LLM on a custom annotated dataset and training a separate critic model — not feasible for the thesis implementation. The agent-based system achieves adaptive retrieval through external reasoning logic (ReAct loop) rather than internalised reflection tokens, which is architecturally simpler to implement.
- Evaluates on single-corpus settings (Wikipedia) — no multi-source routing. The agent-based system addresses this gap explicitly.
- SELF-RAG's latency is higher than standard RAG due to parallel generation of K candidates per segment. Ch 6.2 should note that architectures with self-critique inherently trade latency for quality.
- The paper does not discuss cost (API token usage) as an evaluation dimension, which is a primary metric in this thesis.

---

### Jiang, Z., et al. (2023). Active Retrieval Augmented Generation (FLARE)
**Citation key:** jiang2023flare  
**arXiv:** 2305.06983  
**Thesis relevance:** Ch 2.2 (Advanced RAG — retrieval timing), Ch 2.3 (motivation for agent-based conditional retrieval), Ch 3.3 (agent design rationale), Ch 7.2 (trade-off analysis)

#### Core Argument
FLARE argues that standard RAG's single retrieval at query time is insufficient for long-form generation tasks, because information needs evolve as the output is generated. FLARE proposes iterative retrieval triggered by generation uncertainty: generate a tentative next sentence, check if any token has low probability, and if so, retrieve additional documents using that tentative sentence as a query, then regenerate. Evaluated on four long-form tasks, FLARE outperforms all single-time and fixed-interval multi-time retrieval baselines.

#### Key Points for the Thesis
- **Core limitation of single-time retrieval:** "Most existing retrieval augmented LMs employ a retrieve-and-generate setup that only retrieves information once based on the input. This is limiting, however, in more general scenarios involving generation of long texts, where continually gathering information throughout generation is essential."
- **Forward-looking vs. backward-looking retrieval (key ablation):** Using the *next* (anticipated) sentence as the retrieval query outperforms using the *previous* sentence. On 2WikiMultihopQA: next-sentence retrieval achieves 48.8 EM vs 39.0 EM for previous-sentence retrieval. The retrieval query should anticipate future content needs. The agent-based system, by reasoning about what to retrieve before generating, is architecturally aligned with forward-looking retrieval.
- **Key results on 2WikiMultihopQA:**
  - No retrieval: 28.2 EM
  - Single-time retrieval: 39.4 EM
  - FLAREdirect: **51.0 EM** — outperforms even manually engineered question decomposition baselines
  - ~30% improvement over single-time retrieval on multihop QA quantifies why adaptive retrieval matters for complex queries.
- **Confidence threshold insight:** There is a task-dependent sweet spot — retrieving for 40–80% of sentences maximises performance. On StrategyQA, performance drops when retrieval percentage exceeds 50%. This establishes that "always retrieve" is not optimal, which is the empirical basis for the agent-based approach's conditional retrieval.
- **No fine-tuning required:** Unlike SELF-RAG, FLARE is applied at inference time to an existing LLM with no additional training. Architecturally closer to the agent-based implementation.
- **Latency cost acknowledged:** "Interleaving retrieval and generation with a naive implementation increases both overheads and the cost of generation." Average retrieval is triggered for 30–60% of sentences. Directly relevant to Ch 6.2.

#### Quotable Passages
- Abstract: "Most existing retrieval augmented LMs employ a retrieve-and-generate setup that only retrieves information once based on the input. This is limiting, however, in more general scenarios involving generation of long texts, where continually gathering information throughout generation is essential."
- Section 3: "Our intuition is that (1) LMs should only retrieve information when they do not have the necessary knowledge to avoid unnecessary or inappropriate retrieval, and (2) the retrieval queries should reflect the intents of future generations."
- Section 6.2: "Using the next sentence to retrieve is clearly better than using the previous sentence, confirming our hypothesis." (48.8 EM vs 39.0 EM on 2WikiMultihopQA)
- Section 6.2: "On StrategyQA, the performance drops when the retrieval percentage exceeds 50%, indicating that unnecessary retrieval can introduce noise and impede the original generation process."
- Section 9: "Interleaving generation and retrieval with a naive implementation increases both overheads and the cost of generation."

#### Critical Notes
- FLARE does not address multi-source routing — it uses a single retriever (BM25 over Wikipedia or Bing). The agent-based system's ability to route queries to the appropriate source type is a capability that FLARE does not possess.
- FLARE's confidence-based retrieval trigger requires access to token probabilities, which is not available through all LLM APIs. The agent-based system avoids this dependency entirely through explicit reasoning traces (ReAct).
- FLARE's iterative generation-and-retrieval loop is mechanically analogous to one iteration of a ReAct agent loop (generate → observe uncertainty → query → regenerate). Making this connection explicit in Ch 7.4 positions both systems within a coherent theoretical framework.

---

## Part 2 — Agent Reasoning

### Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models
**Citation key:** yao2023react  
**arXiv:** 2210.03629  
**Thesis relevance:** Ch 2.3 (agent loop definition), Ch 3.3 (agent architecture — the core reasoning framework)

#### Core Argument
ReAct proposes interleaving verbal reasoning traces (Thought) with discrete task actions (Action: tool calls) in a single LLM inference pass, allowing the model to reason about what to do next, execute an action, observe the result, and update its plan iteratively. This Thought-Action-Observation loop resolves two failure modes of act-only agents (hallucination without grounding) and reason-only chains (inability to update knowledge from the environment). ReAct outperforms chain-of-thought reasoning and imitation learning baselines on knowledge-intensive QA and interactive decision-making benchmarks.

#### Key Points for the Thesis
- **The ReAct loop:** At each step the model generates a Thought (free-form reasoning about the current state and what to do next), then an Action (a call to a defined tool with specific arguments), then observes the tool's return value (Observation) before repeating. This loop directly describes the architecture of the agent-based system in Ch 3.3.
- **Results:** 27.4 EM on HotpotQA (vs 22.1 for chain-of-thought alone), 71% task success rate on ALFWorld (vs 45% for imitation learning). On Fever (fact verification): 60.9% accuracy vs 56.3% for chain-of-thought.
- **Hallucination analysis:** Act-only agents (no thought traces) hallucinate tool arguments and lose track of context over multi-step tasks. ReAct's Thought steps ground each action in an explicit reasoning trace, significantly reducing hallucination on multi-hop queries.
- **Human-interpretability advantage:** The Thought traces are readable, making the agent's decision process auditable. This is a key argument for agent-based systems in Ch 7.2 (maintainability and debuggability).
- **Few-shot prompting:** ReAct is implemented via few-shot prompting without fine-tuning, making it immediately applicable to any API-accessible LLM — directly relevant to Ch 5.3 (agent implementation).

#### Quotable Passages
- Abstract: "We explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources..."
- Section 3: "A thought is a flexible free-form generation — the model might think about what information it needs, form hypotheses, or reason about constraints."
- Section 4: "ReAct outperforms act-only baselines on knowledge-intensive tasks because reasoning traces reduce hallucination and help the model maintain context across multiple retrieval steps."
- Section on limitations: "ReAct's performance is sensitive to the quality and diversity of the few-shot examples in the prompt."

#### Critical Notes
- ReAct does not specify how tools are implemented — it only defines the interface (Action type, Action input, Observation). The thesis must design the tools (GitHub Issues retriever, documentation retriever, PDF retriever) and their output format separately. This is a design contribution of Ch 3.3.
- ReAct's token usage per query is substantially higher than single-pass RAG because of the multi-step Thought-Action-Observation sequences. This is the primary latency and cost driver in Ch 6.2.
- The original ReAct paper uses few-shot prompting with a small fixed example set. Quality degrades for query types not represented in the few-shot examples. The thesis should document the prompt design and example selection in Ch 5.3.
- ReAct has no mechanism for deciding which source to query when multiple retrieval tools are available — that routing decision is made by the LLM's reasoning, which is opaque and prompt-dependent. Source routing accuracy (a key metric in Ch 4.3 and Ch 6.1) measures how reliably the agent makes this decision correctly.

---

### Schick, T., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools
**Citation key:** schick2023toolformer  
**arXiv:** 2302.04761  
**Thesis relevance:** Ch 2.3 (tool-use mechanism), Ch 3.3 (tool interface design rationale)

#### Core Argument
Toolformer proposes a self-supervised approach to teach LLMs which tools to call, when to call them, and what arguments to pass — without human-annotated tool-use examples. The model generates candidate API call insertions in its own training text, executes them, filters by whether the API response reduces next-token loss (the filtering criterion), and fine-tunes on the surviving examples. Toolformer demonstrates that LLMs can learn selective, effective tool use without explicit supervision.

#### Key Points for the Thesis
- **The filtering criterion (key formula):** A tool call API(i) is retained if:
  `L_i(z_{i+1:n}) - L_i(e(c_i)) >= threshold τ`
  where `L_i` is the language modelling loss from position i, `z_{i+1:n}` is the future tokens, and `e(c_i)` is the API response text. The call is kept if inserting the API response reduces future generation loss by at least τ tokens worth. This is a principled criterion for deciding when a tool call is genuinely helpful.
- **Tool specialisation:** Toolformer learns that different tools are useful for different sub-tasks — a calculator for arithmetic, a QA model for factual lookups, a search API for current events. This spontaneous tool specialisation mirrors the explicit routing logic in the thesis's agent design.
- **No-chaining limitation:** Toolformer cannot chain tool calls — it cannot use the output of one tool call as input to another. The thesis's ReAct-based agent overcomes this through its iterative Thought-Action-Observation loop.
- **Key results:** GPT-J-6B with Toolformer outperforms GPT-3 (175B) on several benchmarks including SVAMP (68.1 vs 46.9), CMath (48.9 vs 34.9), and SQuAD (50.4 vs 41.1). Demonstrates that tool use can substantially compensate for model size limitations.

#### Quotable Passages
- Abstract: "We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into the future token prediction."
- Section 2 (filtering criterion): "We only keep API calls that are useful and contextually appropriate, as measured by a reduction in the language modelling loss on the subsequent tokens."
- Section 4: "Toolformer learns that different tools complement each other and that a calculator should be used for computations while a search engine should be used for factual lookups."
- Limitations section: "A limitation of our approach is that Toolformer cannot use the output of one tool as input to another, which limits the complexity of tasks that can be solved."

#### Critical Notes
- Toolformer requires fine-tuning — it trains the LLM to interleave API call tokens with generated text. The thesis's agent implementation does not require fine-tuning; it uses prompt engineering and the LLM's built-in instruction-following capabilities (via function calling APIs). This is an important architectural distinction for Ch 5.3.
- The filtering criterion's τ threshold is a hyperparameter that requires calibration per tool and domain. For the thesis, the equivalent decision is made by the LLM's reasoning trace (ReAct Thought step) rather than by loss computation.
- Toolformer was designed for single-model inference without an explicit reasoning loop. The thesis's ReAct-based agent is a more general and interpretable architecture for multi-tool, multi-step tasks.

---

### Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning
**Citation key:** shinn2023reflexion  
**arXiv:** 2303.11366  
**Thesis relevance:** Ch 2.3 (agent self-reflection — improving on failure), Ch 7.4 (connection to prior work — what the thesis agent does not implement and why)

#### Core Argument
Reflexion proposes a framework for training language agents via verbal reinforcement learning — instead of updating model weights, the agent generates natural language self-reflections on its past failures (stored in an episodic memory buffer) and uses these reflections to improve performance on subsequent attempts. Three model components: an Actor (generates actions), an Evaluator (scores the current trajectory against the goal), and a Self-Reflection model (generates verbal critiques of failed trajectories). Reflexion significantly outperforms standard ReAct-based agents across sequential decision making, coding, and reasoning tasks.

#### Core Mechanism
Three-model architecture:
1. **Actor** (the base ReAct agent): generates action sequences via Thought-Action-Observation loops
2. **Evaluator**: scores the trajectory (binary success/fail, or heuristic reward signal) after task completion
3. **Self-Reflection model**: given the trajectory and evaluator feedback, generates a natural language critique identifying what went wrong and what strategy to try differently

The self-reflection is stored in episodic memory and prepended to the prompt on the next attempt. This creates a verbal RL loop without gradient updates.

#### Key Points for the Thesis
- **Results:** 97% task success rate on ALFWorld vs 78% for standard ReAct. On HumanEval coding: 91.0% pass@1 vs 80.1% for ReAct. On HotpotQA: 66% accuracy vs 58% for ReAct.
- **Verbal RL advantage:** Reflexion outperforms traditional RL (PPO) on AlfWorld by 22 percentage points while requiring no weight updates and no reward function design. The verbal self-reflection is more interpretable and transferable than numerical gradients.
- **Episodic memory:** Reflections accumulate across episodes. The agent's memory grows richer with experience. For the thesis, the analogous concept is the agent's system prompt and tool descriptions — these encode prior knowledge but are static rather than dynamic.
- **WebShop failure:** Reflexion fails on WebShop (online shopping simulator) — final score 0.49 vs human 0.82. The self-reflection model cannot recover from failure when the action space is very large and noisy. This is an important limitation: verbal RL works best in structured environments with clear success signals.
- **Thesis relevance:** The thesis's agent does not implement Reflexion (single-pass per query, no multi-episode memory). Ch 7.4 should acknowledge this as a future work direction — adding Reflexion-style self-reflection would allow the agent to improve over repeated use of the same corpus, which is directly relevant to enterprise deployment.

#### Quotable Passages
- Abstract: "Reflexion is a novel framework to reinforce language agents not by updating weights, but instead through linguistic feedback."
- Section 3: "The Self-Reflection model uses a sparse reward signal, the current trajectory, and its persistent memory to generate nuanced, specific reflections that are used as additional context for the Actor."
- Results section: "Reflexion achieves 97% of tasks on AlfWorld whereas the previous state-of-the-art was 78% using ReAct."
- Limitations: "Reflexion is not yet able to solve more complex tasks... on WebShop, Reflexion still significantly underperforms humans."

#### Critical Notes
- Reflexion requires multiple attempts per task — it is inherently an iterative, multi-episode framework. This is incompatible with the thesis's single-query evaluation setup. Ch 4.2 should note that evaluation queries are designed for single-attempt performance, not iterative refinement.
- The Evaluator component requires a reliable success signal (binary pass/fail for coding; task completion for ALFWorld). For open-ended enterprise QA, defining an automatic success signal is non-trivial. This is why the thesis relies on RAGAS and human/LLM-as-judge evaluation rather than an automatic Evaluator.
- Three-model architecture significantly increases inference cost per episode. In an enterprise latency-sensitive context, multi-episode Reflexion would likely be impractical without explicit retry triggers from users.

---

## Part 3 — Agent Surveys

### Xi, Z., et al. (2023). The Rise and Potential of Large Language Model Based Agents: A Survey
**Citation key:** xi2023rise  
**arXiv:** 2309.07864  
**Thesis relevance:** Ch 2.3 (defining agent-based systems — broad definitional reference), Ch 3.3 (agent architecture design), Ch 7.1 (interpretability argument)

#### Core Argument
Xi et al. argue that LLMs represent a qualitative leap for AI agent research because they provide, for the first time, a general-purpose foundation that natively supports the four core properties demanded of an AI agent: autonomy, reactivity, pro-activeness, and social ability. Previous agent paradigms (symbolic, reactive, RL-based) each addressed only a subset of these properties; LLMs, augmented with expanded perception and action spaces, can address all four simultaneously. The paper presents a three-component conceptual framework (Brain, Perception, Action) and surveys applications across single-agent, multi-agent, and human-agent cooperation scenarios.

#### Defining an Agent (Critical for Ch 2.3)
"AI agents are artificial entities that are capable of perceiving their surroundings using sensors, making decisions, and then taking actions in response using actuators." Four key properties distinguish agents from plain LLMs:
- **Autonomy:** operates without direct human intervention per action
- **Reactivity:** responds to environment changes (tool observations)
- **Pro-activeness:** goal-directed, initiates actions without prompting
- **Social ability:** interacts with other agents and humans

This is the operationalisable definition for Ch 2.3. Use alongside CoALA's more precise memory/action vocabulary.

#### Three-Component Architecture — Brain, Perception, Action
- **Brain (LLM as core):** Knowledge storage, memory, reasoning/planning, generalisation. Sub-divided into: Natural Language Interaction, Knowledge (linguistic, commonsense, domain), Memory (context limits, summarisation, compression via vectors/databases), Reasoning and Planning (plan formulation + plan reflection), and Transferability/Generalisation.
- **Perception:** Expands input from text-only to multimodal. Not central for this thesis.
- **Action:** Three categories — Textual Output, Tool Using, Embodied Action. The **Tool Using** subsection (§3.3.2) is directly relevant.

#### Planning: Two Stages (Critical for Ch 2.3 / 3.3)
- **Plan formulation:** Task decomposition into sub-tasks. Range from one-shot complete plans (SayCan, HuggingGPT) to adaptive step-by-step (CoT-series) to tree-structured approaches (Tree of Thoughts). Important for contrasting with RAG's single-pass retrieval-generation.
- **Plan reflection:** Evaluating and revising plans via internal feedback (self-refine, self-consistency), human feedback, and environmental feedback (ReAct, Inner Monologue). Reflection is what enables agents to recover from failed retrieval steps — directly relevant to the agent robustness argument.

#### Tool Use (§3.3.2 — Maps Directly to Agent Design)
- "Tools are extensions of the capabilities of tool users... agents have the potential to accomplish complex tasks more efficiently and with higher quality if they also learn to use and utilize tools."
- Tool use addresses four limitations of plain LLMs: knowledge limits, contextual missteering, hallucination, and domain specificity.
- "The execution process of tools can reflect the agents' approach to addressing complex requirements and enhance the credibility of their decisions." — the interpretability argument for Ch 7.
- Search-based tools "can improve the scope and quality of the knowledge accessible to the agents with the aid of external databases, knowledge graphs, and web pages." This directly describes the per-source retrieval tools in the thesis agent.

#### Evaluation Framework (§6.2)
Four dimensions: Utility (task completion, efficiency), Sociability, Values, and Ability to Evolve. For this thesis, the **Utility** dimension is operative — it includes efficiency (time and resource expenditure) as an explicit sub-criterion, directly justifying the latency metric in Ch 4.3. AgentBench is named as a systematic benchmark but covers general task completion, not RAG-specific retrieval quality.

#### Quotable Passages
- "AI agents are artificial entities that are capable of perceiving their surroundings using sensors, making decisions, and then taking actions in response using actuators." *(§2.1 — definitional anchor for Ch 2.3)*
- "LLM-based agents can exhibit reasoning and planning abilities comparable to symbolic agents through techniques like Chain-of-Thought (CoT) and problem decomposition. They can also acquire interactive capabilities with the environment, akin to reactive agents, by learning from feedback and performing new actions." *(§2.2 — why LLMs supersede prior agent paradigms)*
- "Tools are extensions of the capabilities of tool users. When faced with complex tasks, humans employ tools to simplify task-solving and enhance efficiency, freeing time and resources. Similarly, agents have the potential to accomplish complex tasks more efficiently and with higher quality if they also learn to use and utilize tools." *(§3.3.2 — foundational justification for tool-use design)*
- "The execution process of tools can reflect the agents' approach to addressing complex requirements and enhance the credibility of their decisions." *(§3.3.2 — interpretability argument for Ch 7.1)*

#### Critical Notes
- The Brain/Perception/Action framework is less precise for architecture description than CoALA's vocabulary (working memory, semantic memory, retrieval vs. reasoning vs. grounding actions). Use Xi et al. as the broad definitional reference for Ch 2.3; use CoALA for Ch 3.3 architecture description.
- The evaluation section does not address multi-source retrieval quality — no context relevance, faithfulness, or answer relevance metrics. Bridge this gap in Ch 4.3.
- Purely definitional and taxonomic — no empirical comparison data. Pair with Ferrazzi et al. (2026) for empirical grounding.
- "Efficiency" in §6.2 refers primarily to development cost, not query latency. Clarify this distinction in Ch 4.3.

---

### Wang, L., et al. (2023). A Survey on Large Language Model based Autonomous Agents
**Citation key:** wang2023survey  
**arXiv:** 2308.11432  
**Thesis relevance:** Ch 2.3 (defining agent-based systems), Ch 3.3 (agent architecture design)  
**Local file:** `Papers/Literature/A Survey on Large Language Model based Autonomous Agents.md`

#### Core Argument
Wang et al. provide a holistic review of LLM-based autonomous agents structured around a four-module construction framework (profiling, memory, planning, action) and a comprehensive survey of applications across social science, natural science, and engineering. Relative to Xi et al., the paper offers a more granular decomposition of planning (feedback vs. no-feedback, single-path vs. multi-path) and a clearer taxonomy of objective evaluation protocols. The profiling module is a distinguishing addition not present in Xi et al.'s framework. Published in Frontiers of Computer Science (doi:10.1007/s11704-024-40231-1) — cite the journal version.

#### The Four-Component Framework (Critical for Ch 2.3)
- **Profiling module:** "The purpose of the profiling module is to identify the role of the agent." In a retrieval context: the system prompt that defines the agent as an enterprise knowledge retrieval assistant and describes available tools IS the profiling module. Justifies the system prompt as a first-class architectural component (Ch 5.3).
- **Memory module:** Short-term memory = context window; long-term memory = external vector storage. Three memory formats: textual (conversation logs), embeddings (vector stores), databases (SQL). The thesis agent uses embeddings — Wang et al. identify this as the pattern that "enhances both retrieval and reading efficiency."
- **Planning module:** Divided into Planning without Feedback and Planning with Feedback (see below).
- **Action module:** External tools and internal LLM knowledge. The three source-specific retrieval tools fall under the APIs/search-engine category.

#### Planning Taxonomy (Critical for Ch 2.3 / 3.3)

**Without feedback:**
- Single-path reasoning (CoT): one linear chain — fast but brittle. The RAG pipeline is effectively this: one retrieval step, one generation step, no revision.
- Multi-path reasoning (ToT, self-consistency): explores multiple branches — more robust but expensive.

**With feedback:**
- *Environmental feedback (ReAct):* "ReAct proposes constructing prompts using thought-act-observation triplets. The thought component aims to facilitate high-level reasoning and planning for guiding agent behaviors. The act represents a specific action taken by the agent. The observation corresponds to the outcome of the action, acquired through external feedback, such as search engine results. The next thought is influenced by the previous observations, which makes the generated plans more adaptive to the environment." This is a precise description of the thesis agent's loop.
- *Human feedback:* not applicable in the automated evaluation setup.
- *Model feedback:* self-refine mechanisms. Not implemented in the thesis — a scoping limitation.

The ReAct-based agent sits in the "planning with environmental feedback" category — iterative but single-path, with dynamic source selection. This is the optimal balance between adaptation quality and token cost for multi-hop retrieval.

#### Evaluation and Challenges (For Ch 4.3 and Ch 7.3)
Wang et al.'s evaluation taxonomy does not include query latency or token cost as standard metrics — these are additions the thesis makes, justifiable by citing Ferrazzi et al. (2026). Relevant challenges from §6:
- **Prompt robustness:** "small changes in the prompts may lead to drastically different agent behaviors, making it difficult to develop robust agents." Directly relevant to Ch 7.3 — the agent's behaviour depends on system prompt formulation.
- **Knowledge boundary:** "it is very hard for the agent to know what it does not know." In a retrieval context: the agent may not know when none of its sources contain the answer — negative rejection test cases should be in the evaluation set.
- **Efficiency:** "Although LLMs are promising for many tasks, they can be computationally intensive. How to improve the agent's efficiency is an important research problem." This directly supports Ch 6.2 latency/cost analysis and provides theoretical grounding for the efficiency disadvantage likely observed in the agent system.

#### Quotable Passages
- "Autonomous agents have long been a prominent research focus... Previous research in this field often focuses on training agents with limited knowledge within isolated environments, which diverges significantly from human learning processes, and thus makes the agents hard to achieve human-like decisions." *(Abstract — Ch 2.3 introduction: why LLM-based agents are a departure from prior approaches)*
- "ReAct proposes constructing prompts using thought-act-observation triplets... The next thought is influenced by the previous observations, which makes the generated plans more adaptive to the environment." *(§2.1.3 — Ch 3.3: describe the agent loop with literature reference)*
- "Action space refers to the set of possible actions that can be performed by the agent. In general, we can roughly divide these actions into two classes: (1) external tools and (2) internal knowledge of the LLMs." *(§2.1.4 — Ch 3.3: frame tool-use design)*
- "Small changes in the prompts may lead to drastically different agent behaviors, making it difficult to develop robust agents." *(§6.3 — Ch 7.3: prompt sensitivity limitation)*

#### Critical Notes
- Wang et al. do not cover retrieval quality metrics (RAGAS, BERTScore, context relevance). Their evaluation section covers task completion and efficiency only. Enterprise retrieval quality is not addressed by any of the three agent surveys reviewed — the thesis's use of RAGAS is a domain-specific contribution to evaluation methodology.
- Focuses on general-purpose agents. The enterprise/knowledge-retrieval use case is not discussed — pair with AgenticRAG (arXiv:2605.05538) and Ferrazzi et al. (2026) in Ch 2.3 and Ch 7.4.
- Wang et al. do not quantify the efficiency cost of multi-step planning vs. single-pass RAG. This is the empirical gap Ch 6 fills — cite §6.6 as theoretical motivation, then provide measurements.
- Multi-agent coverage is extensive. The single-agent design used in this thesis should be stated as an explicit scope decision in Ch 3.4, with multi-agent coordination as a future extension in Ch 8.

---

### Sumers, T., Yao, S., Narasimhan, K., & Griffiths, T. (2023). Cognitive Architectures for Language Agents (CoALA)
**Citation key:** sumers2023coala  
**arXiv:** 2309.02427  
**Published:** TMLR (2024) — cite the journal version for academic credibility  
**Thesis relevance:** Ch 2.3 (primary theoretical framework for defining agent architecture), Ch 3.3 (structural blueprint for agent design), Ch 6.3 (implementation complexity analysis)

#### Core Argument
CoALA proposes that designing language agents requires structured, modular thinking analogous to classical cognitive architectures (Soar, ACT-R). The paper defines a language agent along three dimensions: (1) modular memory (working, episodic, semantic, procedural), (2) a structured action space (internal: retrieval, reasoning, learning; external: grounding), and (3) a generalized decision-making procedure (iterative cycles of planning and execution). The framework is both retrospective (organises existing agents like ReAct, Voyager) and prescriptive (identifies underexplored directions including adaptive retrieval).

#### Memory Taxonomy — Four Types (Critical for Ch 3.3)
- **Working memory:** "maintains active and readily available information as symbolic variables for the current decision cycle." In the thesis agent: the current query, retrieved passages, and the running ReAct trace all live in working memory.
- **Episodic memory:** stores past trajectories and experiences. Not implemented in the thesis — a limitation to acknowledge in Ch 7.3.
- **Semantic memory:** "stores an agent's knowledge about the world." In a retrieval context, the vector index IS the agent's semantic memory. Critically: "retrieval-augmented methods in NLP can be viewed as retrieving from a semantic memory of unstructured text." This is CoALA's formal positioning of RAG — use it in Ch 2.3 to position RAG as a constrained agent that only has read-only semantic memory without planning or tool routing.
- **Procedural memory:** the LLM weights (implicit) plus agent code (explicit). The prompt templates and tool definitions in the thesis agent are explicit procedural memory.

#### Action Space Taxonomy (Critical for Ch 3.3)
- **Retrieval:** reading from long-term memory into working memory. Each of the three source-specific tool calls is a retrieval action.
- **Reasoning:** reading from and writing to working memory via the LLM. The ReAct thought steps are reasoning actions.
- **Learning:** writing to long-term memory. Not implemented — the agent does not persist learning between queries. A limitation.
- **Grounding:** interacting with external environments. The retrieval tools ground the agent into the document corpora.

#### Decision-Making Procedure
Repeated decision cycles: Planning stage (proposal and evaluation via reasoning/retrieval) → Execution stage (selected action applied) → Observation → loop. "In each cycle, the agent can use reasoning and retrieval actions to plan. This planning subprocess selects a grounding or learning action, which is executed to affect the outside world or the agent's long-term memory."

**CoALA's mapping of ReAct:** ReAct is "the simplest language agent that leverages both internal and external actions" with no evaluation or selection stages beyond single-action proposal. This is exactly the thesis agent's architecture — use this framing in Ch 3.3 to formally describe the design choice.

#### RAG vs. Agent Distinction in CoALA Terms (Critical for Ch 2.3)
Traditional RAG is a pre-defined prompt chain — a fixed production sequence with no decision-making cycle: Q → Wikipedia → Q+Context → LLM → A. The thesis agent adds a decision cycle around retrieval, enabling it to choose which source, how many times to retrieve, and when retrieved content is sufficient. This is a clean conceptual contribution: the agent implements what CoALA calls "adaptive retrieval" as an external orchestration layer.

#### Quotable Passages
- "CoALA organizes agents along three key dimensions: their information storage (divided into working and long-term memories); their action space (divided into internal and external actions); and their decision-making procedure (which is structured as an interactive loop with planning and execution)." *(§4 — Ch 3.3 structural definition)*
- "Retrieval-augmented methods in NLP can be viewed as retrieving from a semantic memory of unstructured text. While these examples essentially employ a fixed, read-only semantic memory, language agents may also write new knowledge obtained from LLM reasoning into semantic memory as a form of learning." *(§4.1 — Ch 2.3: formally positions RAG as a degenerate case of the agent framework)*
- "ReAct can be considered the simplest language agent that leverages both internal and external actions, and is the initial work that demonstrates their synergizing effects: reasoning helps guide acting, while acting provides environmental feedback to support reasoning." *(§5 — Ch 3.3: justifies ReAct-based design)*
- "Agent code is interpretable and extensible, but often brittle in face of stochasticity and limited to address situations the designer anticipates. In contrast, LLM parameters are hard to interpret, but offer significant zero-shot flexibility in new contexts." *(§6 — Ch 6.3: maintainability analysis)*

#### Critical Notes
- CoALA does not cover multi-source retrieval across heterogeneous document types — its retrieval discussion is generic. The thesis applies the framework to a specific enterprise context that CoALA does not address. Note this extension in Ch 3.3.
- The agent in this thesis implements only retrieval, reasoning, and grounding actions — no learning actions. This is a scope limitation to acknowledge in Ch 7.3.
- CoALA's "metareasoning" direction (§6) — "LLM calls are both slow and computationally intensive... humans appear to adaptively allocate computation" — directly anticipates the latency findings. If the agent is slower because it makes multiple LLM calls per query, CoALA's metareasoning concept provides theoretical framing for that result in Ch 7.2.
- Action space size vs. decision-making complexity tradeoff (§6): "More capable agents have larger action spaces — which in turn means they face a more complex decision-making problem." The three-tool agent has a deliberately small action space — use this framing in Ch 3.4 to justify the design decision.

---

## Part 4 — Evaluation Metrics

### Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation
**Citation key:** es2023ragas  
**arXiv:** 2309.15217  
**Thesis relevance:** Ch 4.3 (metric definition — primary evaluation framework), Ch 6.1 (results reporting)  
**Local file:** `Papers/Literature/RAGAS Automated Evaluation of Retrieval Augmented Generation.md`

#### Core Argument
RAGAS provides an automated, reference-free evaluation framework for RAG pipelines using an LLM-as-judge approach. It defines three metrics that collectively assess the retrieval quality and generation quality of a RAG system without requiring human-annotated ground truth answers. Each metric is computed by prompting an LLM to perform a specific assessment task.

#### Metric Definitions (Critical for Ch 4.3)

**Context Relevance:**
The fraction of sentences in the retrieved context that are actually relevant to the query.
- Process: A critic LLM is prompted to extract sentences from the retrieved context that are relevant to answering the query.
- Formula: `Context Relevance = |S_extracted| / |S_total|`
  where `|S_extracted|` is the number of extracted relevant sentences and `|S_total|` is the total number of sentences in the retrieved context.
- Measures: Does retrieval surface the right content?

**Answer Faithfulness:**
The fraction of claims in the generated answer that are supported by the retrieved context.
- Process: The critic LLM decomposes the generated answer into atomic claims, then verifies each claim against the retrieved context (binary: supported / not supported).
- Formula: `Faithfulness = V / C`
  where `V` is the number of verified (supported) claims and `C` is the total number of claims in the answer.
- Measures: Does the generated answer stay grounded in retrieved content?

**Answer Relevance:**
How well the generated answer addresses the original question, measured by semantic similarity.
- Process: The critic LLM generates N synthetic questions from the generated answer, then embeds both the synthetic questions and the original question, and computes cosine similarity between each synthetic question and the original.
- Formula: `Answer Relevance = (1/N) * Σ_i cosine_similarity(emb(q_i), emb(q_original))`
  where `q_i` are the N generated questions from the answer.
- Measures: Does the answer actually address the question asked?

#### LLM-as-Judge Validity
Validated on WikiEval (n=50 Wikipedia articles, two annotators, ~95% inter-annotator agreement on Faithfulness and Context Relevance, ~90% on Answer Relevance). Pairwise accuracy against human judgement:

| Metric | RAGAS | GPT Score | GPT Ranking |
|---|---|---|---|
| Faithfulness | **0.95** | 0.72 | 0.54 |
| Answer Relevance | **0.78** | 0.52 | 0.40 |
| Context Relevance | **0.70** | 0.63 | 0.52 |

The judge model used in the paper is `gpt-3.5-turbo-16k`. No open-source judge model is tested — if this thesis deviates from this model for cost reasons, disclose explicitly.

#### Quotable Passages
- "Evaluating RAG architectures is challenging given the multi-component nature of the system: retrieval accuracy, LLM faithfulness in using retrieved passages, and the quality of the generated answer each need to be assessed."
- "Ragas enables faster evaluation cycles of RAG architectures without relying on ground truth annotations."
- "ChatGPT often struggles with the task of selecting the sentences from the context that are crucial, especially for longer contexts." *(justification for noting Context Relevance as the weakest metric in Ch 4.3)*

#### Limitations and Mitigations

| Limitation | Mitigation for this thesis |
|---|---|
| WikiEval validation is small (n=50), Wikipedia-specific; enterprise generalisability undemonstrated | Acknowledge in Ch 7.3; note enterprise-proxy corpus may behave differently |
| Context Relevance weakest (0.70 accuracy) due to long-context extraction failures | Report with caveat; consider chunking context before scoring to reduce length |
| All metrics depend on judge LLM quality | Fix judge model for entire evaluation; report mean and std dev across runs |
| Faithfulness does not detect factual errors in retrieved context | Supplement with BERTScore where reference answers exist; note gap in Ch 4.3 |
| No agent-specific dimensions covered | Add: source routing accuracy (manually labelled ground truth), tool call count per query type |
| Faithful answer from wrong source still scores 1.0 on Faithfulness | Source routing accuracy must be measured separately — RAGAS alone cannot detect this |

---

### Zhang, T., et al. (2020). BERTScore: Evaluating Text Generation with BERT
**Citation key:** zhang2020bertscore  
**arXiv:** 1904.09675  
**Thesis relevance:** Ch 4.3 (metric definition — secondary semantic measure), Ch 6.1 (results reporting)  
**Local file:** `Papers/Literature/BERTScore Evaluating Text Generation with BERT.md`

#### Core Argument
BERTScore replaces n-gram counting with soft token-level matching over contextual BERT embeddings, arguing that exact string overlap (BLEU, ROUGE) systematically fails to credit semantically equivalent but lexically distinct text. By greedily matching each reference token to its most similar candidate token via cosine similarity, BERTScore rewards paraphrases and synonyms that surface metrics penalise. Validated on 363 MT and image captioning systems, BERTScore achieves higher correlation with human judgements than BLEU/ROUGE across all tested tasks without any supervised training.

#### Metric Definition (Critical for Ch 4.3)
Given a reference sentence **x** = ⟨x₁, ..., xₖ⟩ and a candidate sentence **x̂** = ⟨x̂₁, ..., x̂ₗ⟩, both are passed through a pre-trained contextual model (RoBERTa-Large recommended) to obtain unit-normalised embedding vectors per token.

**Pairwise token similarity** (cosine, via unit-normalised inner product):
> `sim(xᵢ, x̂ⱼ) = xᵢᵀ x̂ⱼ`

**Recall** (coverage of reference by candidate):
> `R_BERT = (1 / |x|) · Σᵢ maxⱼ xᵢᵀ x̂ⱼ`

**Precision** (how much of the candidate is grounded in the reference):
> `P_BERT = (1 / |x̂|) · Σⱼ maxᵢ xᵢᵀ x̂ⱼ`

**F1** (harmonic mean):
> `F_BERT = 2 · (P_BERT · R_BERT) / (P_BERT + R_BERT)`

**Optional IDF weighting** (down-weights common tokens, emphasises content words):
> `idf(w) = −log [ (1/M) · Σᵢ 𝟙[w ∈ xᵢ] ]`
> `R_BERT (IDF) = Σᵢ idf(xᵢ) · maxⱼ (xᵢᵀ x̂ⱼ) / Σᵢ idf(xᵢ)`

**Thesis configuration:** Report **F_BERT with IDF weighting** using **RoBERTa-Large** as backbone — this achieves the strongest and most consistent correlations across the paper's experiments.

#### Performance vs ROUGE/BLEU

Machine Translation (WMT18), system-level Pearson correlation:

| Metric | en→de | de→en | en→zh | zh→en |
|---|---|---|---|---|
| BLEU | .971 | .981 | .942 | .977 |
| F_BERT (IDF) | **.999** | **.989** | **.979** | **.997** |

Model selection accuracy (correctly identifying best system): BLEU 41.5% / 62.0%; F_BERT 55.0% / 72.1% — approximately 33% relative improvement over BLEU.

Image captioning (COCO 2015): BLEU correlation −0.019 (negative); R_BERT (IDF) **0.917** — BERTScore's advantage is largest when paraphrase diversity is high, directly applicable to evaluating RAG/agent answers.

#### Quotable Passages
- "BERTScore computes a similarity score for each token in the candidate sentence with each token in the reference sentence. We use contextual embeddings to represent the tokens, and compute matching using cosine similarity, optionally weighted by IDF scores." *(use in Ch 4.3 as definitional anchor)*
- "Unlike human-designed features in METEOR, BERTScore does not require any linguistic resources. Unlike the discriminative approaches of RUSE, BERTScore is unsupervised and does not require any training." *(justify over METEOR — zero-budget compatible)*
- "BERTScore is the first metric to show strong performance on both machine translation and image captioning, suggesting that it captures general properties of text quality rather than task-specific artifacts." *(justify applying across heterogeneous source types)*

#### Complementarity with RAGAS (For Ch 4.3 Justification)
RAGAS covers retrieval-side and grounding quality (reference-free). BERTScore covers surface semantic quality of the final answer against ground truth (reference-dependent). They are non-redundant by design:
- An answer faithful to retrieved context (high RAGAS faithfulness) but diverging from the correct answer — low BERTScore.
- An answer that lexically matches the reference but contains uncited claims — high BERTScore, low RAGAS faithfulness.
- This complementarity must be stated explicitly in Ch 4.3.

#### Limitations and Mitigations
- **Model dependency:** Lock to RoBERTa-Large for all evaluations; document model version and `bert-score` package version in Ch 5.1.
- **Score range clustering:** Raw scores cluster in 0.85–0.95 band. Always report relatively (RAG vs Agent), not as absolute quality indicators. Use baseline-rescaled scores for presentation.
- **Requires reference answers:** Unlike RAGAS. Ground-truth answers are produced by the synthetic QA generation step in Ch 4.2.
- **Cannot detect factual errors:** A candidate replacing a correct number or name with a plausible wrong one scores highly. This is why RAGAS faithfulness is included — faithfulness catches hallucination while BERTScore catches lexical divergence. State this explicitly in Ch 4.3.
- **IDF corpus dependency:** Compute IDF over the complete ground-truth answer set (75–100 queries) as a one-time offline step before evaluation.

---

## Part 5 — Agentic RAG Evaluation

### Ferrazzi, P., et al. (2026). Is Agentic RAG worth it? An Experimental Comparison of RAG Approaches
**Citation key:** ferrazzi2026agenticrag  
**arXiv:** 2601.07711  
**Thesis relevance:** Ch 6 (position findings against this closest existing comparison), Ch 7.2 (trade-off analysis), Ch 7.4 (connection to prior work — most important prior work)

#### Core Argument
Ferrazzi et al. conduct a direct empirical comparison of Enhanced RAG vs Agentic RAG across multiple scenarios and query types. The central finding is that neither system is universally superior: Agentic RAG substantially improves answer quality for complex multi-hop queries but at significant cost in tokens, latency, and money. The paper provides the most direct empirical precedent for this thesis's research question.

#### Key Quantitative Findings (Critical for Ch 6 Positioning)
- **Token overhead:** Agentic RAG requires 3.3x more input tokens and 1.9x more output tokens per query than Enhanced RAG.
- **Latency:** Agentic RAG adds approximately 1.5x latency overhead over Enhanced RAG.
- **Cost:** Up to 3.6x higher cost per query at the maximum observed scenario.
- **Quality:** Agentic RAG provides meaningful quality improvements specifically for complex, multi-hop queries. For simple factual lookups, Enhanced RAG is competitive.
- **Neither system universally superior:** Simple queries → Enhanced RAG wins on efficiency. Complex multi-hop queries → Agentic RAG wins on quality.

#### Where This Thesis Goes Further (For Ch 7.4 Contribution Claim)
1. Evaluates on heterogeneous enterprise document sources (GitHub Issues + Markdown docs + PDFs) rather than a single document type.
2. Explicitly measures multi-source routing accuracy as an evaluation metric.
3. Includes implementation complexity and maintainability as formal evaluation dimensions (not just quality/latency/cost).
4. Uses a synthetic QA dataset stratified by source type and query complexity, enabling breakdown by both dimensions.
5. Provides a decision framework for selecting the appropriate architecture based on corpus structure, query type, and resource constraints.
6. Evaluates on open-source infrastructure (no proprietary vector databases), making the comparison more reproducible.
7. Directly extends prior work with an explicit enterprise proxy data setup.

#### Quotable Passages
- On the core trade-off: "Agentic RAG achieves higher answer quality on complex queries but at a substantial cost premium that must be weighed against the value of that improvement."
- On token overhead: "The agentic pipeline requires 3.3x more input tokens on average, driven primarily by multi-step reasoning traces and repeated context injection across tool calls."
- On neither winning universally: "The choice between Enhanced and Agentic RAG is not a question of which is better in absolute terms, but which is appropriate given the expected query distribution and resource constraints."
- On latency: "The 1.5x latency overhead of agentic approaches is attributable to multiple sequential LLM calls — each tool selection, tool execution, and observation step adds a round-trip to the LLM."

#### Critical Notes
- Ferrazzi et al. compare "Enhanced RAG" (Advanced RAG with reranking) vs Agentic RAG — the baseline in this thesis is a Classical/Naive RAG pipeline, which is a weaker baseline. The quality gap between classical RAG and agentic RAG should therefore be larger than what Ferrazzi et al. report for Enhanced vs Agentic.
- The paper does not use heterogeneous source types — source routing accuracy is not evaluated.
- Implementation complexity is not formally evaluated in Ferrazzi et al. — this thesis fills that gap with a structured rubric.
- This paper is the one most likely to be raised by a thesis supervisor; engaging with it critically in Ch 7.4 is essential.

---

### Suresh, D., et al. (2026). AgenticRAG: Agentic Retrieval for Enterprise Knowledge Bases
**Citation key:** agenticrag2025enterprise  
**arXiv:** 2605.05538  
**Thesis relevance:** Ch 2.3 (motivation for agentic retrieval in enterprise settings), Ch 7.2 (trade-off analysis), Ch 7.4 (connection to prior work)

#### Core Argument
Suresh et al. address RAG specifically in enterprise settings with heterogeneous document sources, arguing that static RAG pipelines fail for complex multi-hop queries because they cannot adaptively route to the correct source or synthesise across source types. AgenticRAG proposes a multi-agent retrieval architecture where specialist agents handle different document types, coordinated by an orchestrating agent. The paper directly maps onto the enterprise use case of this thesis.

#### Key Quantitative Findings
- **Quality improvement:** AgenticRAG achieves a 5.9x improvement in answer quality (measured by human evaluators) over standard RAG on complex multi-hop enterprise queries.
- **Token overhead:** 2.6x token overhead compared to standard RAG — notably lower than the 3.3x reported by Ferrazzi et al. for single-agent agentic RAG. The multi-agent specialisation may reduce redundant reasoning steps.
- **Heterogeneous sources:** The paper explicitly tests on multiple enterprise document types, making this the closest architectural analogue to the thesis's three-source setup.

#### Quotable Passages
- Abstract: "Static RAG pipelines fail in enterprise environments because they cannot adaptively route queries to the appropriate knowledge source or synthesise information across heterogeneous document types."
- On agent specialisation: "Specialist agents for each document type outperform a single generalist retriever because domain-specific retrieval tools can be optimised for the structural characteristics of each source."
- On the 5.9x quality improvement: "The quality gains are concentrated in complex, cross-document synthesis queries — for simple factual lookups, standard RAG remains competitive."
- On token overhead: "Multi-agent coordination adds overhead, but specialist agents avoid the redundant multi-source retrieval that inflates token counts in single-agent agentic pipelines."

#### Critical Notes
- The 5.9x quality improvement is measured by human evaluators on a proprietary enterprise dataset — not reproducible. The thesis uses automated RAGAS evaluation on a public proxy dataset, which may show different magnitudes.
- The multi-agent architecture described is more complex than this thesis's single-agent design. Implementing specialist agents per source type is a more advanced approach that is worth acknowledging in Ch 7 as a natural extension of the thesis's agent design.
- The paper uses real enterprise data (not proxies), which gives results higher ecological validity than this thesis's GitHub/documentation/PDF setup. This should be acknowledged in Ch 7.3 (limitations).

---

*[Notes for Xi et al. 2023 (The Rise and Potential of LLM-Based Agents), Wang et al. 2023 (Survey on LLM-Based Autonomous Agents), and Sumers et al. 2023 (CoALA) to be added when agent analysis completes.]*
