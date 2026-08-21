# BERTScore: Evaluating Text Generation with BERT

**Authors:** Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi (Cornell University and ASAPP Inc.)
**arXiv:** 1904.09675
**Published:** ICLR 2020

---

## Abstract

BERTScore is an automatic evaluation metric for text generation that computes token similarity using contextual embeddings from pre-trained BERT models rather than exact string matching. Testing on 363 machine translation and image captioning systems, BERTScore correlates better with human judgments than existing metrics and demonstrates stronger model selection performance. The metric is unsupervised, requires no external linguistic resources, and supports 104 languages via multilingual BERT variants.

---

## 1. Introduction

Automatic text generation evaluation metrics such as BLEU and ROUGE rely on exact n-gram matching, which fails to account for synonyms, paraphrases, and flexible word ordering. For example, the candidate "people like foreign cars" and the reference "consumers prefer imported cars" convey the same meaning but share no n-gram overlap. BERTScore addresses this by leveraging contextual embeddings, where token representations encode both meaning and sentence context, enabling soft semantic matching.

---

## 2. Metric Definition

### 2.1 Token Representation

For a reference sentence **x** = ⟨x₁, x₂, ..., xₖ⟩ and a candidate sentence **x̂** = ⟨x̂₁, x̂₂, ..., x̂ₗ⟩, both are tokenized and passed through a pre-trained BERT model to obtain contextual embedding vectors. Each token xᵢ is represented by its contextual embedding **xᵢ** ∈ ℝᵈ, normalized to unit length so that cosine similarity reduces to the inner product:

> cos(xᵢ, x̂ⱼ) = xᵢᵀ x̂ⱼ

### 2.2 Greedy Token Matching

Token similarity between the reference and candidate is computed via greedy matching: each token in one sentence is matched to the most similar token in the other.

**BERTScore Recall** (how well the reference is covered by the candidate):

> R_BERT = (1 / |x|) · Σᵢ maxⱼ xᵢᵀ x̂ⱼ

**BERTScore Precision** (how much of the candidate is supported by the reference):

> P_BERT = (1 / |x̂|) · Σⱼ maxᵢ xᵢᵀ x̂ⱼ

**BERTScore F1** (harmonic mean of precision and recall):

> F_BERT = 2 · (P_BERT · R_BERT) / (P_BERT + R_BERT)

### 2.3 IDF Importance Weighting (Optional)

Rare tokens are down-weighted by IDF scores computed from the reference corpus of M sentences:

> idf(w) = −log [ (1/M) · Σᵢ 𝟙[w ∈ x⁽ⁱ⁾] ]

The IDF-weighted recall becomes:

> R_BERT = Σᵢ idf(xᵢ) · maxⱼ (xᵢᵀ x̂ⱼ) / Σᵢ idf(xᵢ)

Precision and F1 are similarly extended. IDF weighting reduces the influence of common tokens (e.g., function words) that carry little semantic content.

### 2.4 Baseline Rescaling

Raw BERTScore values cluster in a limited range rather than spanning [0, 1]. To improve human readability without affecting ranking ability, scores are rescaled using an empirical lower bound *b* computed from random segment pairs:

> R̂_BERT = (R_BERT − b) / (1 − b)

This rescaling preserves rank ordering and thus does not affect evaluation validity.

---

## 3. Experimental Results

### 3.1 Machine Translation — WMT18 Benchmark

System-level Pearson correlation with human judgements across language pairs:

| Metric | en→de | de→en | en→zh | zh→en |
|--------|-------|-------|-------|-------|
| BLEU | .971 | .981 | .942 | .977 |
| F_BERT | .999 | .989 | .979 | .997 |

Segment-level Kendall τ correlation (en→de / de→en):

- BLEU: .969 / .977
- F_BERT: .998 / .988

Model selection accuracy (fraction of times best system is correctly identified):

- BLEU: .415 / .620 (en→de / de→en)
- F_BERT with IDF: .550 / .721

BERTScore outperforms BLEU and ROUGE across all language pairs at both system and segment level. It approaches RUSE, a supervised metric trained with human judgements — without requiring any labelled data.

### 3.2 Image Captioning — COCO 2015

Correlation of evaluation metrics with human image quality judgements (M1 metric):

- BLEU: −0.019
- METEOR: 0.606
- SPICE (task-specific): 0.759
- R_BERT (IDF-weighted): 0.917
- Leic (task-specific, image-aware): 0.939

R_BERT substantially outperforms BLEU and METEOR and is competitive with task-specific metrics that have access to image information.

### 3.3 Robustness on Adversarial Paraphrases (PAWS)

On the adversarial PAWS-QQP dataset, which contains paraphrase pairs that differ only in word order, BERTScore achieved AUC of 0.693, demonstrating robustness to surface-level variation where n-gram metrics fail near-completely.

---

## 4. Key Properties

**What BERTScore captures:**
- Semantic similarity through soft token matching (synonyms, paraphrases)
- Word order sensitivity via contextual embeddings that encode positional context
- Cross-lingual evaluation (104 languages via mBERT)
- Rare-token weighting via optional IDF scores
- Fully differentiable — can serve as a training loss

**What BERTScore does NOT capture:**
- Factual correctness: high similarity scores can be assigned to plausible-sounding but factually wrong text (e.g., swapping "German" for "English" in a translation)
- Numerical precision: "5ft 11in" and "1.80 metres" receive high similarity despite different values
- Attribution/grounding: BERTScore does not assess whether an answer is grounded in a retrieved context — this is covered by RAGAS faithfulness
- Retrieval quality: BERTScore evaluates candidate text against a reference string; it has no notion of what was retrieved — this is RAGAS context relevance
- Answer relevance to question intent: RAGAS answer relevance covers this; BERTScore does not evaluate the question-answer relationship

---

## 5. Limitations

1. **Model dependency:** Performance varies significantly with the choice of BERT variant. RoBERTa-Large differs substantially from BERT-Base. No single model configuration outperforms all others across all tasks.

2. **Domain sensitivity:** On low-resource languages or specialized domains not well represented in BERT pre-training data, embedding quality degrades and metric reliability decreases.

3. **Score range clustering:** Raw scores cluster in a narrow band, making absolute score values hard to interpret without baseline rescaling. Rescaled scores are more readable but may obscure small differences.

4. **IDF corpus dependency:** IDF weighting requires access to the full reference corpus at test time to compute document frequencies. Benefits over unweighted BERTScore are task-dependent and not consistently positive.

5. **Factual accuracy blindspot:** BERTScore cannot detect factual errors that are semantically proximate to the reference. This is a fundamental limitation: it is a surface-semantic metric, not a factual-verification metric.

6. **Computational cost:** Requires GPU for practical throughput (192.5 pairs/second on GPU). For large-scale evaluations, this must be accounted for in the experimental setup.

---

## 6. Quotable Passages

> "BERTScore computes a similarity score for each token in the candidate sentence with each token in the reference sentence. We use contextual embeddings to represent the tokens, and compute matching using cosine similarity."

> "Unlike human-designed features in METEOR, BERTScore does not require any linguistic resources. Unlike the discriminative approaches of RUSE, BERTScore is unsupervised and does not require any training."

> "BERTScore is the first to show strong performance on both machine translation and image captioning, suggesting that it captures general properties of text quality rather than task-specific artifacts."

> "The key advantage over BLEU is that BERTScore captures synonymy and paraphrasing, which are widespread in natural language and systematically ignored by n-gram matching."

---

## 7. Recommended BERT Model for Thesis Use

The paper recommends **RoBERTa-Large** for English-language evaluation as it consistently achieves highest correlations with human judgements across tasks. For practical thesis use with API-based generation outputs, `roberta-large` via the `bert-score` Python package is the standard choice.

```python
from bert_score import score
P, R, F1 = score(candidates, references, lang="en", model_type="roberta-large")
```
