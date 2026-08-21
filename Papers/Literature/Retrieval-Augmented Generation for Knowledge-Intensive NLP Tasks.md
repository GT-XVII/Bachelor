# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela

**Year:** 2020 (NeurIPS 2020)

**arXiv ID:** 2005.11401

**Submitted:** 22 May 2020 (v1); last revised 12 Apr 2021 (v4)

**DOI:** https://doi.org/10.48550/arXiv.2005.11401

---

## Abstract

Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge remain open research problems. Pre-trained models with a differentiable access mechanism to explicit non-parametric memory can overcome this issue, but have so far been only investigated for extractive downstream tasks. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models which combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. We compare two RAG formulations, one which conditions on the same retrieved passages across the whole generated sequence, the other can use different passages per token. We fine-tune and evaluate our models on a wide range of knowledge-intensive NLP tasks and set the state-of-the-art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures. For language generation tasks, we find that RAG models generate more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline.

---

## 1. Introduction

Pre-trained neural language models have been shown to learn a substantial amount of in-depth knowledge from data. They can do so without any access to an external memory, as a parameterized implicit knowledge base. While this development is exciting, such models do have downsides: They cannot easily expand or revise their memory, can't straightforwardly provide insight into their predictions, and may produce "hallucinations."

Hybrid models that combine parametric memory with non-parametric (i.e., retrieval-based) memories can address some of these issues because knowledge can be directly revised and expanded, and accessed knowledge can be inspected and interpreted.

Here, the authors bring hybrid parametric and non-parametric memory to the "workhorse of NLP," i.e. sequence-to-sequence (seq2seq) models. They endow pre-trained, parametric-memory generation models with a non-parametric memory through a general-purpose fine-tuning approach referred to as retrieval-augmented generation (RAG).

RAG models are built where the parametric memory is a pre-trained seq2seq transformer, and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. The retriever provides latent documents conditioned on the input, and the seq2seq model then conditions on these latent documents together with the input to generate the output.

Like T5 or BART, RAG can be fine-tuned on any seq2seq task, whereby both the generator and retriever are jointly learned. The paper explores a setting where both parametric and non-parametric memory components are pre-trained and pre-loaded with extensive knowledge. Crucially, by using pre-trained access mechanisms, the ability to access knowledge is present without additional training.

The results highlight the benefits of combining parametric and non-parametric memory with generation for knowledge-intensive tasks — tasks that humans could not reasonably be expected to perform without access to an external knowledge source.

---

## 2. Methods

### 2.1 Overview

RAG models use the input sequence x to retrieve text documents z and use them as additional context when generating the target sequence y. The models leverage two components:

1. **A retriever** with parameters η that returns (top-K truncated) distributions over text passages given a query x.
2. **A generator** parametrized by θ that generates a current token based on a context of the previous i−1 tokens.

To train the retriever and generator end-to-end, the retrieved document is treated as a latent variable. Two models are proposed that marginalize over the latent documents in different ways to produce a distribution over generated text.

### 2.2 RAG-Sequence Model

In RAG-Sequence, the model uses the same document to predict each target token. Formally, the model uses a single retrieved document to generate the complete output sequence, and then marginalizes across documents.

### 2.3 RAG-Token Model

In RAG-Token, the model can predict each target token based on a different document. This allows the generator to use different retrieved passages for different parts of the generated answer.

### 2.4 Retriever: Dense Passage Retrieval (DPR)

The retriever component is based on Dense Passage Retrieval (DPR), using a bi-encoder architecture with BERT_BASE encoders. The document encoder is kept fixed during fine-tuning; only the query encoder and the BART generator are updated.

### 2.5 Generator: BART

The generator is BART-large, a pre-trained seq2seq transformer with 400M parameters. Input is constructed by concatenating the query with each retrieved document, and BART generates the output conditioned on this input.

### 2.6 Training

The models are trained end-to-end by minimizing the negative marginal log-likelihood of targets, treating retrieved documents as latent variables. No explicit retrieval supervision is used.

---

## 3. Experiments

### 3.1 Knowledge Source

Following prior work, the December 2018 Wikipedia dump is used. Each Wikipedia article is split into disjoint 100-word chunks, producing a total of 21M documents. These are encoded using the DPR document encoder and indexed using FAISS for efficient maximum inner product search.

### 3.2 Tasks

Experiments are conducted on a wide range of knowledge-intensive tasks:

- **Open-Domain QA:** Natural Questions (NQ), TriviaQA, WebQuestions, CuratedTrec
- **Abstractive QA:** MS-MARCO NLG
- **Jeopardy Question Generation**
- **Fact Verification:** FEVER (3-way and 2-way classification)

### 3.3 Decoding

For QA tasks, greedy decoding is used. For generation tasks, beam search with a beam size of four is used.

---

## 4. Results

### 4.1 Open-Domain QA (Exact Match)

| Dataset       | RAG-Token | RAG-Sequence |
|---------------|-----------|--------------|
| Natural Questions | 44.1  | 44.5         |
| TriviaQA      | 55.2      | 56.8         |
| WebQuestions  | 66.1      | 68.0         |
| CuratedTrec   | 45.5      | 45.2         |

On all four open-domain QA tasks, RAG sets a new state of the art, outperforming both parametric seq2seq models (T5, BART) and task-specific retrieve-and-extract architectures.

### 4.2 Generation Tasks

**MS-MARCO NLG:**

| Model         | BLEU-1 | ROUGE-L |
|---------------|--------|---------|
| BART baseline | 38.2   | 41.6    |
| RAG-Token     | 40.1   | 41.5    |
| RAG-Sequence  | 40.8   | 44.2    |

RAG-Sequence outperforms BART on Open MS-MARCO NLG by 2.6 BLEU points and 2.6 ROUGE-L points.

**Jeopardy Question Generation:**

| Model     | Q-BLEU-1 |
|-----------|----------|
| BART      | 19.7     |
| RAG-Token | 22.2     |

### 4.3 Fact Verification (FEVER)

RAG achieved within 4.3% of state-of-the-art pipeline models on 3-way classification, and within 2.7% on 2-way classification, despite not using retrieval supervision.

### 4.4 Human Evaluation

Human evaluators assessed factuality and specificity on Jeopardy question generation:

- RAG was rated more factual than BART in **42.7%** of cases.
- BART was rated more factual than RAG in only **7.1%** of cases.
- RAG-Sequence's generations are more diverse than RAG-Token's.

---

## 5. Related Work

Prior work has shown that retrieval improves performance across a variety of NLP tasks. A single, pre-trained language model has been shown to achieve strong performance on various classification tasks. The RAG framework distinguishes itself from prior work by using fully pre-trained, end-to-end learnable components rather than task-specific retrieval architectures, and by applying the approach to generative (rather than purely extractive) tasks.

---

## 6. Discussion

The RAG models obtain state-of-the-art results on open-domain QA. The retrieval index can be "hot-swapped" to update the model's knowledge without requiring any retraining. This property is important for production systems where knowledge must be kept current. The paper also shows that RAG generates less hallucinated content than purely parametric baselines, with improved factuality and specificity in human evaluations.

Compared to T5-11B (11 billion parameters), RAG achieves comparable or better performance with only 626M parameters (400M BART + 220M DPR query encoder), demonstrating the efficiency advantage of externalizing knowledge to a retrieval index rather than encoding it entirely in model weights.

---

## References

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems (NeurIPS 2020)*. arXiv:2005.11401.

Key referenced works:
- Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. (DPR)
- Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text.
- Lee et al. (2019). Latent Retrieval for Weakly Supervised Open Domain Question Answering.
