"""
Full RAG pipeline: query → retrieve → generate → log.

Usage:
    from rag.pipeline import RAGPipeline
    rag = RAGPipeline()
    result = rag.query("What are the CNCF recommendations for supply chain security?")
    print(result["answer"])
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .retrieve import Retriever
from .generate import Generator
from . import config


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._log_path = config.LOG_DIR / f"rag_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.jsonl"

    def query(self, question: str) -> dict:
        t_start = time.perf_counter()

        # ── HyDE: generate hypothesis before retrieval ─────────────────────────
        hypothesis = None
        t_hyde = 0.0
        if config.USE_HYDE:
            t_hyde_start = time.perf_counter()
            hypothesis = self.generator.generate_hypothesis(question)
            t_hyde = time.perf_counter() - t_hyde_start

        # ── Retrieval ──────────────────────────────────────────────────────────
        t_retrieve_start = time.perf_counter()
        chunks = self.retriever.retrieve(question, hypothesis=hypothesis)
        t_retrieve = time.perf_counter() - t_retrieve_start

        # ── Generation ────────────────────────────────────────────────────────
        t_generate_start = time.perf_counter()
        gen = self.generator.generate(question, chunks)
        t_generate = time.perf_counter() - t_generate_start

        t_total = time.perf_counter() - t_start

        result = {
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "system":         "rag",
            "query":          question,
            "answer":         gen["answer"],
            "finish_reason":  gen["finish_reason"],
            # HyDE hypothesis used for retrieval (None if USE_HYDE=False)
            "hyde_hypothesis": hypothesis,
            # retrieved context provenance
            "retrieved": [
                {
                    "chunk_id":     c["chunk_id"],
                    "source_type":  c["source_type"],
                    "doc_id":       c["doc_id"],
                    "title":        c["title"],
                    "faiss_score":  c.get("faiss_score"),
                    "rerank_score": c.get("rerank_score"),
                }
                for c in chunks
            ],
            # latency (seconds) — hyde_s is 0.0 when USE_HYDE=False
            "latency": {
                "hyde_s":     round(t_hyde,     4),
                "retrieve_s": round(t_retrieve, 4),
                "generate_s": round(t_generate, 4),
                "total_s":    round(t_total,    4),
            },
            # token usage
            "tokens": {
                "input":  gen["input_tokens"],
                "output": gen["output_tokens"],
                "total":  gen["input_tokens"] + gen["output_tokens"],
            },
        }

        self._log(result)
        return result

    def _log(self, result: dict):
        with open(self._log_path, "a") as f:
            f.write(json.dumps(result) + "\n")
