"""Central configuration — all tuneable parameters in one place."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "data" / "corpus"
INDEX_DIR  = BASE_DIR / "agent" / "indexes"
LOG_DIR    = BASE_DIR / "agent" / "logs"

CORPUS_FILES = {
    "github_issue":   CORPUS_DIR / "github_issues.jsonl",
    "markdown_doc":   CORPUS_DIR / "markdown_docs.jsonl",
    "pdf_whitepaper": CORPUS_DIR / "pdf_whitepapers.jsonl",
}

# Per-source index paths (one FAISS index per enterprise proxy source type)
INDEX_PATHS = {
    "github_issue":   INDEX_DIR / "github_issues.faiss",
    "markdown_doc":   INDEX_DIR / "markdown_docs.faiss",
    "pdf_whitepaper": INDEX_DIR / "pdf_whitepapers.faiss",
}
CHUNKS_PATHS = {
    "github_issue":   INDEX_DIR / "github_issues_chunks.jsonl",
    "markdown_doc":   INDEX_DIR / "markdown_docs_chunks.jsonl",
    "pdf_whitepaper": INDEX_DIR / "pdf_whitepapers_chunks.jsonl",
}

# Markdown corpus is downsampled to balance index sizes across sources.
# None = no limit.
SOURCE_LIMITS = {
    "github_issue":   None,
    "markdown_doc":   400,
    "pdf_whitepaper": None,
}

# ── AI Refinery (OpenAI-compatible) ───────────────────────────────────────────
AI_REFINERY_BASE_URL = os.getenv("AI_REFINERY_BASE_URL", "https://api.airefinery.accenture.com/v1")
AI_REFINERY_API_KEY  = os.getenv("AI_REFINERY_API_KEY", "")
LLM_MODEL            = os.getenv("LLM_MODEL", "gpt-4o")

# sdk_version header required by the AI Refinery API gateway
LLM_DEFAULT_HEADERS = {"sdk_version": "1.31.3"}

LLM_TEMPERATURE = 0.0   # deterministic for reproducible evaluation
LLM_MAX_TOKENS  = 2048  # 1024 was too tight for gpt-oss-120b: reasoning tokens consumed budget before visible answer

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM        = 768
EMBEDDING_BATCH_SIZE = 64
BGE_QUERY_PREFIX     = "Represent this sentence for searching relevant passages: "

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE_WORDS    = 400   # identical to RAG pipeline — shared data layer
CHUNK_OVERLAP_WORDS = 60

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 20   # FAISS candidates before reranking (per tool call)
TOP_K_FINAL     = 5    # chunks returned per tool call after reranking

# ── Reranking ─────────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Agent loop ────────────────────────────────────────────────────────────────
MAX_ITERATIONS = 15  # empirically calibrated: gpt-oss-120b uses up to 9 tool calls per query

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a precise enterprise knowledge assistant with access to three document sources: "
    "GitHub issue tracker records, technical documentation, and CNCF whitepapers. "
    "Use the available tools to search for relevant information before answering. "
    "If a question spans multiple sources, search each relevant source. "
    "Base your final answer only on retrieved evidence and cite source titles where relevant. "
    "If after searching the available sources you cannot find relevant information, "
    "state clearly that the answer is not available in the knowledge base — do not continue searching."
)
