"""
LangGraph StateGraph for the multi-agent retrieval system.

Architecture:
    START
      └─► retrieve_all_sources  (sequential FAISS retrieval across all 3 sources)
      └─► synthesize             (one LLM call with all retrieved chunks)
      └─► END

Sub-agents are pure FAISS retrievers — no LLM call per source. The synthesise
node makes exactly one LLM call, keeping total token consumption and latency
comparable to the RAG pipeline while always querying all three source types.

Note on parallelism: Apple Silicon MPS does not support concurrent inference
from multiple Python threads. Retrieval therefore runs sequentially across
sources. Since FAISS retrieval takes only ~100ms per source, the dominant
latency term remains the single LLM synthesis call — the key improvement over
the single-agent ReAct loop (up to 15 sequential LLM calls).
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from agent.tools import SourceRetriever
from . import config


# ── State ─────────────────────────────────────────────────────────────────────

class MultiAgentState(TypedDict):
    query:          str
    source_results: list[dict]   # one entry per source, populated by retrieve_all_sources
    answer:         str
    finish_reason:  str


# ── Shared retriever instances (loaded once at graph build time) ───────────────

_retrievers: dict[str, SourceRetriever] = {}


def _get_retriever(source: str) -> SourceRetriever:
    if source not in _retrievers:
        _retrievers[source] = SourceRetriever(source)
    return _retrievers[source]


def _format_chunks(chunks: list[dict], source: str) -> str:
    if not chunks:
        return f"No results found in {source}."
    parts = []
    for i, c in enumerate(chunks, 1):
        source_url = c.get("metadata", {}).get("url") or c["doc_id"]
        parts.append(
            f"[{i}] ({c['source_type']}) {c['title']}\n"
            f"Source: {source_url}\n"
            f"Score: {c.get('rerank_score', 0):.3f}\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ── Node: retrieve_all_sources ─────────────────────────────────────────────────

def retrieve_all_sources(state: MultiAgentState) -> dict:
    """
    Sequential FAISS + cross-encoder retrieval across all three source types.
    Returns all source results in one state update.
    """
    source_results = []
    for src in config.ALL_SOURCES:
        retriever = _get_retriever(src)
        chunks = retriever._retrieve(state["query"])
        source_results.append({
            "source":      src,
            "chunks":      chunks,
            "observation": _format_chunks(chunks, src),
        })
    return {"source_results": source_results}


# ── Node: synthesize ───────────────────────────────────────────────────────────

def synthesize(state: MultiAgentState) -> dict:
    """
    Single LLM call: synthesise all retrieved evidence into a final answer.
    Context is bounded — 3 sources × 5 chunks max, no accumulation.
    """
    llm = ChatOpenAI(
        base_url=config.AI_REFINERY_BASE_URL,
        api_key=config.AI_REFINERY_API_KEY,
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        default_headers=config.LLM_DEFAULT_HEADERS,
    )

    context_parts = []
    for result in state["source_results"]:
        label = result["source"].replace("_", " ").title()
        context_parts.append(f"=== {label} ===\n{result['observation']}")
    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=config.SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Question: {state['query']}\n\n"
            f"Retrieved evidence:\n{context}"
        )),
    ]

    response = llm.invoke(messages)

    content = response.content
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    answer = (content if isinstance(content, str) else str(content)).strip()

    finish_reason = "stop"
    if hasattr(response, "response_metadata"):
        finish_reason = response.response_metadata.get("finish_reason", "stop")

    return {"answer": answer, "finish_reason": finish_reason}


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph():
    """
    Build and compile the multi-agent StateGraph.

    All three SourceRetriever instances are pre-loaded sequentially here so
    that FAISS indexes and sentence-transformers models are in memory before
    the graph executes any node.
    """
    print("Loading retrievers…")
    for src in config.ALL_SOURCES:
        _get_retriever(src)

    builder = StateGraph(MultiAgentState)
    builder.add_node("retrieve_all_sources", retrieve_all_sources)
    builder.add_node("synthesize",           synthesize)
    builder.add_edge(START, "retrieve_all_sources")
    builder.add_edge("retrieve_all_sources", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()
