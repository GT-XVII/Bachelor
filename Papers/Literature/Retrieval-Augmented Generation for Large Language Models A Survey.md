# Retrieval-Augmented Generation for Large Language Models: A Survey

**Authors:** Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang
**arXiv:** 2312.10997
**Submitted:** December 18, 2023 (v1); Last revised March 27, 2024 (v5)
**Category:** Computer Science — Computation and Language (cs.CL)
**Source:** https://arxiv.org/abs/2312.10997

---

## Abstract

Large Language Models demonstrate impressive capabilities yet encounter significant hurdles including hallucination, outdated knowledge, and opaque reasoning processes. Retrieval-Augmented Generation (RAG) emerges as a solution by incorporating knowledge from external databases, enhancing accuracy and credibility for knowledge-intensive tasks and enabling continuous updates with domain-specific information. This comprehensive review examines RAG's evolution through three paradigms — Naive, Advanced, and Modular RAG — scrutinizing retrieval, generation, and augmentation techniques. The paper presents evaluation frameworks and benchmarks while identifying current challenges and future research directions.

---

## Introduction

LLMs achieve remarkable success but struggle with domain-specific and knowledge-intensive tasks, notably producing "hallucinations" when handling queries beyond their training data. RAG enhances LLMs by retrieving relevant document chunks from external knowledge bases through semantic similarity calculations, effectively reducing factually incorrect content generation.

RAG technology evolved in distinct stages. Initially coinciding with the Transformer architecture's rise, RAG focused on enhancing language models during pre-training. ChatGPT's arrival marked a pivotal shift toward leveraging in-context learning capabilities. Subsequent development integrated RAG more deeply with LLM fine-tuning techniques.

---

## RAG Taxonomy

### Naive RAG

The foundational methodology encompasses three core phases:

**Indexing:** Raw data undergoes cleaning and extraction, converting diverse formats into plain text. Text is segmented into smaller chunks, which are encoded into vector representations using embedding models and stored in vector databases.

**Retrieval:** Upon receiving a query, the system encodes it using the same embedding model, calculates similarity scores between query vectors and indexed chunks, and retrieves the top K most similar documents.

**Generation:** The original query and retrieved documents combine into a coherent prompt that the LLM uses to formulate responses.

**Limitations of Naive RAG:**
- Retrieval struggles with precision and recall, selecting misaligned chunks
- Generation models may hallucinate content unsupported by retrieved context
- Single retrievals prove insufficient for complex multi-step reasoning
- The approach lacks mechanisms to refine queries or filter poor-quality retrieved content

---

### Advanced RAG

This paradigm introduces targeted improvements addressing Naive RAG's shortcomings through pre-retrieval and post-retrieval strategies.

**Pre-retrieval Process:**
- Indexing optimization: sliding windows, fine-grained segmentation, metadata incorporation
- Query optimization: rewriting, transformation, and expansion techniques to better align the user query with the indexed content

**Post-Retrieval Process:**
- Reranking retrieved information to prioritize pertinent results and reduce noise
- Context compression to reduce information overload while maintaining relevance, preventing the "lost in the middle" problem where LLMs struggle to utilize information from long retrieved contexts

---

### Modular RAG

This architecture provides enhanced flexibility and adaptability beyond previous paradigms, allowing components to be reconfigured or extended for diverse tasks.

**New Modules:**
- **Search module:** Adapts to specific scenarios with diverse data sources, including search engines, databases, and knowledge graphs
- **Memory module:** Leverages LLM memory to guide retrieval through iterative self-enhancement, using the model's own outputs to refine future retrievals
- **Routing:** Navigates diverse data sources, selecting optimal pathways based on query type and available sources
- **Task Adapter:** Tailors RAG to various downstream tasks through automated prompt retrieval and task-specific configuration

**New Patterns:**
- **Rewrite-Retrieve-Read:** Refines queries through LLM-based rewriting before retrieval occurs
- **Generate-Read:** Replaces traditional retrieval with LLM-generated content where appropriate
- **Hybrid retrieval:** Integrates keyword, semantic, and vector searches for more robust document matching
- **Adaptive retrieval:** Enables autonomous determination of when retrieval is necessary, rather than always querying the knowledge base

---

## Retrieval Components

### Chunking Strategies

The most prevalent approach segments documents into fixed-token chunks (100, 256, 512 tokens). Larger chunks capture more context but generate increased noise and processing costs. Smaller chunks reduce noise but may lack necessary context.

Advanced methods include:
- **Recursive splits and sliding windows:** Enable layered retrieval by merging globally related information across multiple processes
- **Small2Big approach:** Uses sentences as retrieval units while providing surrounding sentences as context to LLMs, balancing semantic completeness with context length
- **Fine-grained segmentation with metadata:** Attaches metadata such as timestamps, source URLs, and document categories to chunks to improve retrieval filtering

### Embedding Models

Retrieval depends on calculating similarity between question and document chunk embeddings. Recent high-performing models include AngIE, Voyage, and BGE, benefiting from multi-task instruction tuning.

**Hybrid Retrieval:** Sparse (BM25) and dense (vector) approaches capture different relevance features. Sparse models enhance zero-shot capabilities of dense retrievers, particularly for queries containing rare entities or technical terminology.

**Fine-tuning Embedding Models:** Particularly valuable in highly specialized domains such as healthcare and law, aligning retrievers with generators through LLM-based supervision signals. Domain-specific fine-tuning consistently improves retrieval quality over general-purpose embedding models.

---

## Evaluation Framework

### Quality Scores

**Context Relevance:** Evaluates precision and specificity of retrieved context, ensuring pertinence while minimizing extraneous content processing costs.

**Answer Faithfulness:** Ensures generated answers remain consistent with retrieved context, avoiding contradictions with the source material.

**Answer Relevance:** Requires generated answers to directly address posed questions, effectively handling core inquiries without irrelevant tangents.

### Required Abilities (Four Evaluation Dimensions)

**Noise Robustness:** Assesses the model's capability to manage question-related documents that lack substantive information — i.e., documents that are topically adjacent but do not actually answer the query. A robust system should neither be misled by noisy documents nor fail to produce an answer solely because some retrieved documents are unhelpful.

**Negative Rejection:** Evaluates the model's discernment in refraining from responding when retrieved documents lack the necessary knowledge to answer the question. The system should recognize when retrieval has failed and decline to generate an unsupported answer, rather than hallucinating.

**Information Integration:** Tests the model's proficiency in synthesizing information from multiple documents to answer complex questions that no single document can address alone. This is particularly relevant for multi-hop queries requiring reasoning across sources.

**Counterfactual Robustness:** Examines the model's ability to recognize and disregard known inaccuracies within retrieved documents, even when those inaccuracies are presented within the retrieved context. A robust system should apply prior knowledge to identify and discount factually incorrect retrieved content.

---

## Conclusion

This survey systematically reviews state-of-the-art RAG methods, delineating its evolution through Naive, Advanced, and Modular paradigms. It identifies central technologies in retrieval, generation, and augmentation components, elucidating their synergies. The paper summarizes current assessment methods covering 26 tasks and nearly 50 datasets, outlining evaluation objectives, metrics, benchmarks, and tools. The survey acknowledges that while RAG has significantly improved LLM performance for knowledge-intensive tasks, challenges remain in handling complex multi-step reasoning, maintaining retrieval quality across heterogeneous sources, and managing the trade-off between retrieval cost and generation quality. Future research directions include more adaptive retrieval mechanisms, improved evaluation standards, and tighter integration between retrieval and generation components.

---

## Bibliographic Reference

Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M., & Wang, H. (2024). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv preprint arXiv:2312.10997*. https://arxiv.org/abs/2312.10997
