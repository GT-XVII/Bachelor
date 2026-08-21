"""
LangChain tool definitions, each backed by a per-source FAISS retriever.

Three tools are exposed — one per enterprise proxy source type. Each wraps
the same two-stage retrieval (FAISS + cross-encoder reranking) used in the
RAG pipeline, ensuring per-call retrieval quality is directly comparable.

Usage:
    from agent.tools import ToolBundle
    bundle = ToolBundle()
    tools = bundle.get_tools()   # list of LangChain tool objects
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.tools import tool

from . import config


class SourceRetriever:
    """
    FAISS + cross-encoder retriever scoped to one corpus source type.

    Accumulates per-call chunk history in _call_history so AgentPipeline
    can reconstruct structured retrieved-chunk records after executor.invoke().
    """

    def __init__(self, source_type: str):
        self.source_type = source_type
        print(f"Loading {source_type} index...")
        self.index    = faiss.read_index(str(config.INDEX_PATHS[source_type]))
        self.chunks   = self._load_chunks(config.CHUNKS_PATHS[source_type])
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.reranker = CrossEncoder(config.RERANKER_MODEL)
        self._call_history: list[list[dict]] = []
        print(f"  {source_type}: {self.index.ntotal} vectors, {len(self.chunks)} chunks.")

    def _load_chunks(self, path) -> list[dict]:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def _embed(self, query: str) -> np.ndarray:
        prefixed = config.BGE_QUERY_PREFIX + query
        return self.embedder.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

    def _retrieve(self, query: str) -> list[dict]:
        query_vec = self._embed(query)
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
        pairs     = [(query, c["text"]) for c in candidates]
        ce_scores = self.reranker.predict(pairs)
        for chunk, ce_score in zip(candidates, ce_scores):
            chunk["rerank_score"] = float(ce_score)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[: config.TOP_K_FINAL]

    def search(self, query: str) -> str:
        """Retrieve, record call history, return formatted observation string."""
        chunks = self._retrieve(query)
        self._call_history.append(chunks)
        if not chunks:
            return f"No results found for: {query}"
        parts = []
        for i, c in enumerate(chunks, 1):
            source = c.get("metadata", {}).get("url") or c["doc_id"]
            parts.append(
                f"[{i}] ({c['source_type']}) {c['title']}\n"
                f"Source: {source}\n"
                f"Score: {c.get('rerank_score', 0):.3f}\n"
                f"{c['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def flush_history(self) -> list[list[dict]]:
        """Return and clear accumulated call history. Called by ToolBundle after invoke()."""
        history = self._call_history.copy()
        self._call_history.clear()
        return history


class ToolBundle:
    """
    Holds all three SourceRetriever instances and exposes LangChain tool objects.
    Instantiated once per AgentPipeline; kept alive so call history is accessible
    after executor.invoke() returns.
    """

    # Maps LangChain tool name → corpus source_type key
    _TOOL_TO_SOURCE = {
        "search_github_issues":   "github_issue",
        "search_markdown_docs":   "markdown_doc",
        "search_pdf_whitepapers": "pdf_whitepaper",
    }

    def __init__(self):
        self._retrievers = {
            "github_issue":   SourceRetriever("github_issue"),
            "markdown_doc":   SourceRetriever("markdown_doc"),
            "pdf_whitepaper": SourceRetriever("pdf_whitepaper"),
        }

    def get_tools(self) -> list:
        """Return list of LangChain tool objects. Call once when building the agent."""
        r = self._retrievers

        @tool
        def search_github_issues(query: str) -> str:
            """Search closed GitHub issues from kubernetes/kubernetes and microsoft/vscode.
            Use for bug reports, feature requests, specific errors, and their resolutions."""
            return r["github_issue"].search(query)

        @tool
        def search_markdown_docs(query: str) -> str:
            """Search Kubernetes, Ansible, and CNCF observability documentation.
            Use for conceptual explanations, how-to guides, configuration references,
            and architectural descriptions."""
            return r["markdown_doc"].search(query)

        @tool
        def search_pdf_whitepapers(query: str) -> str:
            """Search CNCF technical whitepapers on security, storage, serverless,
            and platform engineering. Use for standards, best practices, threat models,
            and maturity frameworks."""
            return r["pdf_whitepaper"].search(query)

        return [search_github_issues, search_markdown_docs, search_pdf_whitepapers]

    def collect_retrieved(self, steps: list) -> list[dict]:
        """
        Match AgentExecutor intermediate_steps with retriever call histories and
        return a flat list of chunk dicts annotated with step number and tool name.

        Called after executor.invoke() returns; flushes call history as a side effect.
        """
        histories = {
            tool_name: self._retrievers[src].flush_history()
            for tool_name, src in self._TOOL_TO_SOURCE.items()
        }
        cursors = {t: 0 for t in self._TOOL_TO_SOURCE}

        retrieved = []
        for step_num, (action, _obs) in enumerate(steps, 1):
            tool_name = action.tool
            if tool_name not in cursors:
                continue
            idx = cursors[tool_name]
            if idx >= len(histories.get(tool_name, [])):
                continue
            for chunk in histories[tool_name][idx]:
                retrieved.append({
                    "step":         step_num,
                    "tool":         tool_name,
                    "chunk_id":     chunk["chunk_id"],
                    "source_type":  chunk["source_type"],
                    "doc_id":       chunk["doc_id"],
                    "title":        chunk["title"],
                    "text":         chunk["text"],
                    "faiss_score":  chunk.get("faiss_score"),
                    "rerank_score": chunk.get("rerank_score"),
                })
            cursors[tool_name] += 1
        return retrieved
