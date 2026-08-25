"""Central configuration for the multi-agent pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR  = BASE_DIR / "multi_agent" / "logs"

# Reuse per-source indexes built by agent/ingest.py — no re-ingestion needed.
_AGENT_INDEX_DIR = BASE_DIR / "agent" / "indexes"

INDEX_PATHS = {
    "github_issue":   _AGENT_INDEX_DIR / "github_issues.faiss",
    "markdown_doc":   _AGENT_INDEX_DIR / "markdown_docs.faiss",
    "pdf_whitepaper": _AGENT_INDEX_DIR / "pdf_whitepapers.faiss",
}
CHUNKS_PATHS = {
    "github_issue":   _AGENT_INDEX_DIR / "github_issues_chunks.jsonl",
    "markdown_doc":   _AGENT_INDEX_DIR / "markdown_docs_chunks.jsonl",
    "pdf_whitepaper": _AGENT_INDEX_DIR / "pdf_whitepapers_chunks.jsonl",
}

ALL_SOURCES = ["github_issue", "markdown_doc", "pdf_whitepaper"]

# ── AI Refinery (OpenAI-compatible) ───────────────────────────────────────────
AI_REFINERY_BASE_URL = os.getenv("AI_REFINERY_BASE_URL", "https://api.airefinery.accenture.com/v1")
AI_REFINERY_API_KEY  = os.getenv("AI_REFINERY_API_KEY", "")
LLM_MODEL            = os.getenv("LLM_MODEL", "gpt-4o")

LLM_DEFAULT_HEADERS = {"sdk_version": "1.31.3"}
LLM_TEMPERATURE     = 0.0
LLM_MAX_TOKENS      = 2048

# ── Embedding & Retrieval (identical to agent) ────────────────────────────────
EMBEDDING_MODEL      = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM        = 768
EMBEDDING_BATCH_SIZE = 64
BGE_QUERY_PREFIX     = "Represent this sentence for searching relevant passages: "

TOP_K_RETRIEVAL = 20
TOP_K_FINAL     = 5
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Synthesis prompt ──────────────────────────────────────────────────────────
SYNTHESIS_SYSTEM_PROMPT = (
    "You are a precise enterprise knowledge assistant. "
    "You have been provided with retrieved excerpts from three document sources: "
    "GitHub issue tracker records, technical documentation, and CNCF whitepapers. "
    "Synthesise the evidence from all provided excerpts into a single coherent answer. "
    "Base your answer only on the retrieved text and cite source titles where relevant. "
    "If the retrieved excerpts do not contain sufficient information to answer the question, "
    "state clearly that the answer is not available in the knowledge base."
)
