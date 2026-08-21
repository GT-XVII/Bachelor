# RAGAS: Automated Evaluation of Retrieval Augmented Generation

**Authors:** Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert
**arXiv:** 2309.15217 (cs.CL)
**Submitted:** September 26, 2023 | **Last revised:** April 28, 2025
**DOI:** https://doi.org/10.48550/arXiv.2309.15217
**License:** CC BY 4.0

---

## Abstract

The paper introduces Ragas, a framework enabling reference-free evaluation of Retrieval-Augmented Generation (RAG) systems. RAG systems are composed of a retrieval module and an LLM-based generation module. Ragas assesses multiple dimensions — retrieval accuracy, LLM faithfulness to retrieved passages, and generation quality — without relying on ground truth annotations. This enables faster evaluation cycles of RAG architectures at scale.

---

## 1. Introduction

RAG systems present a unique evaluation challenge: quality depends on the interaction between two components (retrieval and generation), and human annotation of both is expensive and slow. Ragas addresses this by using LLMs as judges, prompting them to evaluate each component separately and in combination.

The framework is reference-free: it does not require pre-existing ground truth answers. This is critical for enterprise deployment contexts where ground truth is unavailable or expensive to produce.

---

## 2. Metrics

### 2.1 Faithfulness

**Definition:** The degree to which all claims in the generated answer are entailed by (inferable from) the retrieved context.

**Computation:**
1. An LLM is prompted to decompose the generated answer `a(q)` into a set of individual statements `S = {s₁, s₂, ..., sₙ}`.
2. For each statement `sᵢ`, a binary verification function `v(sᵢ, c(q))` is applied, where `c(q)` is the retrieved context. The LLM is prompted: *"determine whether they are supported by the information present in the context. Provide a brief explanation for each statement before arriving at the verdict (Yes/No)."*
3. Let `V` be the set of statements verified as supported. The score is:

```
Faithfulness = |V| / |S|
```

Where `|V|` is the number of supported statements and `|S|` is the total number of statements extracted.

**Range:** [0, 1]. A score of 1.0 means every claim in the answer is grounded in the retrieved context. A score of 0 means no claims are supported.

---

### 2.2 Answer Relevance

**Definition:** The degree to which the generated answer actually addresses the posed question — penalising incomplete or redundant responses.

**Computation:**
1. An LLM is prompted to generate `n` candidate questions `{q̂₁, q̂₂, ..., q̂ₙ}` from the generated answer `a(q)`, simulating the reverse: "what question would this answer be responding to?"
2. Each generated question `q̂ᵢ` and the original question `q` are embedded using a text embedding model (the paper uses `text-embedding-ada-002`).
3. Cosine similarity is computed between the original question embedding and each generated question embedding.
4. The final score is the mean cosine similarity:

```
Answer Relevance = (1/n) * Σ sim(q, q̂ᵢ)    for i = 1..n
```

**Range:** [0, 1]. A high score means the answer is tightly aligned with what was actually asked. Answers that are technically correct but off-topic, vague, or padded will score lower.

**Key property:** This metric does not assess factual correctness — it assesses topical alignment. It should be used together with Faithfulness.

---

### 2.3 Context Relevance

**Definition:** The proportion of the retrieved context that is actually necessary to answer the question — penalising over-retrieval of irrelevant information.

**Computation:**
1. An LLM is prompted to extract from the retrieved context `c(q)` only the sentences that are *crucial* for answering the question `q`.
2. The score is computed as:

```
Context Relevance = |extracted sentences| / |total sentences in c(q)|
```

**Range:** [0, 1]. A score of 1.0 means every retrieved sentence was necessary. A lower score indicates that the retrieval returned significant irrelevant content.

**Known limitation (noted by authors):** ChatGPT "often struggles with the task of selecting the sentences from the context that are crucial, especially for longer contexts." This makes Context Relevance the weakest of the three metrics in terms of reliability, and it achieved the lowest human-agreement alignment in the paper's own validation (accuracy 0.70 vs 0.95 for Faithfulness).

---

## 3. WikiEval Validation Dataset

To validate that RAGAS metrics align with human judgement, the authors constructed the **WikiEval** dataset:

- **Source:** 50 Wikipedia articles covering events after September 2021 (post-knowledge-cutoff for GPT-3.5)
- **Question generation:** ChatGPT prompted to generate one question per article
- **Answer generation:** ChatGPT with the Wikipedia article as context
- **Human annotation:** Two fluent English speakers rated each (question, context, answer) triple on all three metrics
- **Inter-annotator agreement:** ~95% for Faithfulness and Context Relevance, ~90% for Answer Relevance

---

## 4. Experimental Results

Ragas was compared against two baselines — **GPT Score** (direct numerical scoring) and **GPT Ranking** (pairwise ranking) — using pairwise accuracy against human judgements on WikiEval:

| Metric             | Ragas | GPT Score | GPT Ranking |
|--------------------|-------|-----------|-------------|
| Faithfulness       | 0.95  | 0.72      | 0.54        |
| Answer Relevance   | 0.78  | 0.52      | 0.40        |
| Context Relevance  | 0.70  | 0.63      | 0.52        |

Ragas significantly outperforms both baselines across all three metrics. The margin is largest for Faithfulness (0.95 vs 0.72 for GPT Score), confirming that the decompose-then-verify approach is more reliable than holistic LLM scoring.

---

## 5. Implementation Details

- **LLM used for evaluation:** `gpt-3.5-turbo-16k` (via OpenAI API)
- **Embedding model:** `text-embedding-ada-002` (for Answer Relevance similarity)
- **Framework integrations:** LangChain, LlamaIndex
- **Design principle:** Reference-free — no ground truth required

---

## 6. Limitations

- Context Relevance is the least reliable metric (accuracy 0.70); LLMs struggle with long contexts and sentence-level extraction.
- All three metrics are LLM-dependent: scoring variance scales with the judge model's quality and prompt sensitivity.
- The validation dataset (WikiEval, n=50) is small and Wikipedia-specific — generalisability to enterprise document types is not demonstrated.
- Faithfulness and Answer Relevance are computed independently; a high combined score does not guarantee the answer is correct, only that it is grounded and on-topic.
- The framework does not measure factual correctness, completeness against an external knowledge base, or multi-hop reasoning quality.

---

## 7. Conclusion

Ragas provides a practical, reference-free evaluation framework for RAG systems using LLM-as-judge methodology. Its three metrics cover the two pipeline stages (retrieval via Context Relevance; generation via Faithfulness and Answer Relevance). The decompose-then-verify approach for Faithfulness achieves near-human accuracy (0.95) on the WikiEval benchmark. The framework is suitable for automated, large-scale evaluation where human annotation is not feasible.

---

## Thesis Notes

### Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation
**Citation key:** es2023ragas
**Thesis relevance:** Ch 4.3 (metric definition), Ch 6 (results reporting)

---

### Core Argument

RAGAS argues that RAG systems require a dedicated evaluation framework because their quality depends on the interaction between two sub-systems — retrieval and generation — which existing NLP metrics (BLEU, ROUGE, BERTScore) are not designed to disentangle. The framework enables reference-free evaluation using LLMs as judges, making it viable for enterprise contexts where annotated ground truth is unavailable. The core methodological claim is that structured prompting (decompose-then-verify for Faithfulness; reverse question generation for Answer Relevance) is more reliable than direct LLM scoring.

---

### Metric Definitions (Critical — must be precise for Ch 4.3)

#### Context Relevance

An LLM is prompted to identify and extract only those sentences from the retrieved context `c(q)` that are essential for answering query `q`. The metric is computed as the ratio of extracted sentences to the total number of sentences in the retrieved context:

```
CR = |extracted sentences| / |total sentences in c(q)|
```

A score approaching 1.0 indicates a precise retrieval with little extraneous content; a low score indicates the retrieval returned a high proportion of irrelevant passages. This metric evaluates the retrieval component of the pipeline.

#### Answer Faithfulness

The generated answer `a(q)` is first decomposed by an LLM into a discrete set of atomic statements `S = {s₁, ..., sₙ}`. Each statement is then independently verified against the retrieved context `c(q)` by a second LLM prompt that returns a binary Yes/No verdict. The score is:

```
Faithfulness = |V| / |S|
```

where `|V|` is the count of statements verified as supported by the context and `|S|` is the total statement count. A score of 1.0 means the answer is fully grounded; any score below 1.0 indicates hallucination relative to the retrieved context.

#### Answer Relevance

An LLM is prompted `n` times to generate candidate questions `{q̂₁, ..., q̂ₙ}` that the given answer `a(q)` would plausibly respond to. The original question `q` and all generated questions are embedded (the paper uses `text-embedding-ada-002`). The metric is the mean cosine similarity between the original question embedding and each generated question embedding:

```
AR = (1/n) * Σ cos_sim(embed(q), embed(q̂ᵢ))    for i = 1..n
```

A high score indicates the answer is tightly focused on the actual question. Answers that are verbose, off-topic, or incomplete will generate candidate questions that diverge from the original, producing a lower score.

---

### LLM-as-Judge Validity

The authors validate RAGAS against human judgements using the WikiEval dataset (n=50, two annotators, ~95% inter-annotator agreement for Faithfulness and Context Relevance, ~90% for Answer Relevance). Pairwise accuracy against human preference on WikiEval:

- Faithfulness: **0.95** (vs GPT Score 0.72, GPT Ranking 0.54)
- Answer Relevance: **0.78** (vs GPT Score 0.52, GPT Ranking 0.40)
- Context Relevance: **0.70** (vs GPT Score 0.63, GPT Ranking 0.52)

The model used for evaluation in the paper is `gpt-3.5-turbo-16k`. The structured decompose-then-verify approach (rather than holistic scoring) is identified as the key factor in Faithfulness's high agreement. The paper does not test with open-source judge models, which is a relevant gap for this thesis's zero-budget setup.

---

### Key Properties

**What RAGAS captures:**
- Whether retrieved content is relevant to the query (Context Relevance)
- Whether the generated answer is grounded in retrieved content, not hallucinated (Faithfulness)
- Whether the generated answer actually addresses the posed question (Answer Relevance)
- All three without requiring reference answers — reference-free design

**What RAGAS does NOT capture:**
- Factual correctness against an external knowledge base (an answer can be faithful to wrong context)
- Multi-hop reasoning quality or task decomposition success
- Source selection accuracy (whether the agent routed to the right document type)
- Latency, token cost, or implementation complexity
- Completeness — whether the answer covers all relevant aspects of the question
- Performance on multi-document synthesis or cross-source queries
- Agent-specific behaviour (tool call count, routing decisions)

**Gaps directly relevant to this thesis:**
- RAGAS does not measure source routing accuracy, which is a key evaluation dimension for the agent system. This thesis must supplement RAGAS with manually labelled ground-truth source labels per query and compute routing accuracy separately.
- RAGAS does not distinguish between agent and RAG on latency or cost. These must be instrumented independently via API response objects.
- Faithfulness scores an answer relative to the retrieved context — if the agent retrieves from the wrong source, a faithful answer to that wrong context still scores 1.0 on Faithfulness. Context Relevance partially mitigates this but does not fully capture source routing quality.

---

### Quotable Passages

1. **On the core motivation:** "Evaluating RAG architectures is challenging given the multi-component nature of the system: retrieval accuracy, LLM faithfulness in using retrieved passages, and the quality of the generated answer each need to be assessed."

2. **On the reference-free design:** "Ragas [...] enables faster evaluation cycles of RAG architectures without relying on ground truth annotations."

3. **On Context Relevance limitation:** "ChatGPT often struggles with the task of selecting the sentences from the context that are crucial, especially for longer contexts."

*(Note: quotes reconstructed from the arXiv HTML rendering. Verify exact wording against the published PDF before inclusion in the thesis.)*

---

### Limitations and Mitigations

| Limitation | Mitigation for this thesis |
|---|---|
| WikiEval validation is small (n=50) and Wikipedia-specific; enterprise generalisability not demonstrated | Acknowledge in Ch 7.3 (Limitations); note that this thesis's enterprise-proxy corpus may exhibit different metric reliability |
| Context Relevance is the weakest metric (accuracy 0.70 vs human) due to long-context sentence extraction failures | Report Context Relevance scores with a caveat; consider chunking retrieval context before scoring to reduce context length |
| All metrics depend on judge LLM quality — variance scales with model capability | Document which LLM is used as judge (fix it for the entire evaluation); report mean and standard deviation across runs |
| Faithfulness does not detect factual errors — an answer faithful to incorrect retrieved content scores 1.0 | Supplement with BERTScore against reference answers where available; note this gap explicitly in Ch 4.3 |
| RAGAS does not cover agent-specific dimensions (routing, tool use, decomposition) | Add custom metrics: source routing accuracy (manually labelled), tool call count per query type |
| No open-source judge model tested — paper uses gpt-3.5-turbo-16k | If using a different judge model for cost reasons, disclose this and note it as a deviation from the original framework |
