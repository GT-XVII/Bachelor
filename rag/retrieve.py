"""FAISS retrieval + cross-encoder reranking + optional MMR diversification."""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from . import config


class Retriever:
    def __init__(self):
        print("Loading FAISS index...")
        self.index  = faiss.read_index(str(config.INDEX_PATH))
        self.chunks = self._load_chunks()
        self.embedder  = SentenceTransformer(config.EMBEDDING_MODEL)
        self.reranker  = CrossEncoder(config.RERANKER_MODEL)
        print(f"Retriever ready — {self.index.ntotal} vectors, {len(self.chunks)} chunks.")

    def _load_chunks(self) -> list[dict]:
        with open(config.CHUNKS_PATH) as f:
            return [json.loads(l) for l in f if l.strip()]

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a retrieval query using the BGE query prefix."""
        prefixed = config.BGE_QUERY_PREFIX + query
        vec = self.embedder.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype("float32")

    def _embed_passage(self, text: str) -> np.ndarray:
        """Embed a passage without the query prefix (matches document embedding space)."""
        vec = self.embedder.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype("float32")

    def _mmr(self, candidates: list[dict], k: int) -> list[dict]:
        """
        Maximal Marginal Relevance selection over the reranked candidate set.

        Balances relevance (cross-encoder rerank score, normalised to [0,1])
        against diversity (cosine similarity between candidate embedding vectors
        reconstructed from the FAISS IndexFlatIP store via _faiss_idx).

        MMR_LAMBDA = 1.0 → pure relevance (identical to plain top-k).
        MMR_LAMBDA = 0.0 → pure diversity.
        """
        if len(candidates) <= k:
            return candidates

        # Normalise rerank scores to [0, 1] for commensurability with cosine sim
        scores = np.array([c["rerank_score"] for c in candidates], dtype=np.float32)
        s_min, s_max = scores.min(), scores.max()
        norm_scores = (scores - s_min) / (s_max - s_min) if s_max > s_min else np.ones_like(scores)

        # Retrieve stored L2-normalised vectors from FAISS (IndexFlatIP.reconstruct)
        vecs = np.stack([self.index.reconstruct(c["_faiss_idx"]) for c in candidates])

        lam = config.MMR_LAMBDA
        selected: list[int] = []
        remaining: list[int] = list(range(len(candidates)))

        while len(selected) < k and remaining:
            if not selected:
                best = max(remaining, key=lambda i: norm_scores[i])
            else:
                sel_vecs = vecs[selected]
                def mmr_score(i: int, sv=sel_vecs) -> float:
                    return lam * float(norm_scores[i]) - (1.0 - lam) * float((vecs[i] @ sv.T).max())
                best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)

        return [candidates[i] for i in selected]

    def retrieve(self, query: str, hypothesis: str | None = None) -> list[dict]:
        """
        Returns top-k chunks after FAISS search + cross-encoder reranking + MMR.

        If hypothesis is provided (HyDE), the hypothesis text is embedded without
        the BGE query prefix and used for FAISS search in place of the raw query.
        The original query is still used for cross-encoder reranking so relevance
        scoring remains grounded in the actual question.
        """
        # Embed: use hypothesis for FAISS (HyDE) but raw query as fallback
        if hypothesis:
            search_vec = self._embed_passage(hypothesis)
        else:
            search_vec = self._embed_query(query)

        # FAISS: inner product on normalised vectors = cosine similarity
        scores, indices = self.index.search(search_vec, config.TOP_K_RETRIEVAL)
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["faiss_score"] = float(score)
            chunk["_faiss_idx"]  = int(idx)   # temp field for MMR vector lookup
            candidates.append(chunk)

        if not candidates:
            return []

        # Cross-encoder reranking always uses the original query
        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = self.reranker.predict(pairs)
        for chunk, ce_score in zip(candidates, ce_scores):
            chunk["rerank_score"] = float(ce_score)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

        if config.USE_MMR:
            result = self._mmr(candidates, config.TOP_K_FINAL)
        else:
            result = candidates[: config.TOP_K_FINAL]

        # Strip temp field before returning
        for c in result:
            c.pop("_faiss_idx", None)
        return result
