"""FAISS retrieval + cross-encoder reranking."""

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
        # BGE models need a prefix for retrieval queries
        prefixed = config.BGE_QUERY_PREFIX + query
        vec = self.embedder.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype("float32")

    def retrieve(self, query: str) -> list[dict]:
        """
        Returns top-k chunks after FAISS search + cross-encoder reranking.
        Each result dict includes the chunk metadata and retrieval scores.
        """
        query_vec = self._embed_query(query)

        # FAISS: inner product on normalised vectors = cosine similarity
        scores, indices = self.index.search(query_vec, config.TOP_K_RETRIEVAL)
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["faiss_score"] = float(score)
            candidates.append(chunk)

        if not candidates:
            return []

        # Cross-encoder reranking
        pairs  = [(query, c["text"]) for c in candidates]
        ce_scores = self.reranker.predict(pairs)
        for chunk, ce_score in zip(candidates, ce_scores):
            chunk["rerank_score"] = float(ce_score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[: config.TOP_K_FINAL]
