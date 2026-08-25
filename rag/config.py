"""Central configuration — all tuneable parameters in one place."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "data" / "corpus"
INDEX_DIR  = BASE_DIR / "rag" / "indexes"
LOG_DIR    = BASE_DIR / "rag" / "logs"

CORPUS_FILES = {
    "github_issue":    CORPUS_DIR / "github_issues.jsonl",
    "markdown_doc":    CORPUS_DIR / "markdown_docs.jsonl",
    "pdf_whitepaper":  CORPUS_DIR / "pdf_whitepapers.jsonl",
}

INDEX_PATH  = INDEX_DIR / "unified.faiss"
CHUNKS_PATH = INDEX_DIR / "unified_chunks.jsonl"

# ── AI Refinery (OpenAI-compatible) ───────────────────────────────────────────
AI_REFINERY_BASE_URL = os.getenv("AI_REFINERY_BASE_URL", "https://api.airefinery.accenture.com/v1")
AI_REFINERY_API_KEY  = os.getenv("AI_REFINERY_API_KEY", "")
LLM_MODEL            = os.getenv("LLM_MODEL", "gpt-4o")

# sdk_version header required by the AI Refinery API gateway
LLM_DEFAULT_HEADERS = {"sdk_version": "1.31.3"}

# Suppress vLLM-specific param for AI Refinery and local proxies
_skip_extra_body = "airefinery.accenture.com" in AI_REFINERY_BASE_URL or "localhost" in AI_REFINERY_BASE_URL
LLM_EXTRA_BODY   = None if _skip_extra_body else {"chat_template_kwargs": {"enable_thinking": False}}

LLM_TEMPERATURE  = 0.0   # deterministic for reproducible evaluation
LLM_MAX_TOKENS   = 2048  # raised from 1024; gpt-oss-120b uses reasoning tokens before visible answer

# ── HyDE (Hypothetical Document Embeddings) ───────────────────────────────────
# Generates a short hypothetical passage from the query and embeds that instead
# of the raw query. Aligns retrieval with the document embedding space, which
# improves recall on ambiguous and synthesis queries.
USE_HYDE        = True
HYDE_MAX_TOKENS = 512   # enough for reasoning model + 2-3 sentence hypothesis

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM        = 768
EMBEDDING_BATCH_SIZE = 64
# BGE models perform best with this query prefix
BGE_QUERY_PREFIX     = "Represent this sentence for searching relevant passages: "

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE_WORDS    = 400   # target words per chunk
CHUNK_OVERLAP_WORDS = 60    # overlap to preserve context across boundaries

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 20   # FAISS candidates before reranking
TOP_K_FINAL     = 5    # chunks passed to the generator after reranking

# ── Reranking ─────────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── MMR (Maximal Marginal Relevance) ──────────────────────────────────────────
# Applied after cross-encoder reranking to diversify the final top-k set.
# Prevents returning five near-duplicate chunks from the same document section.
USE_MMR    = True
MMR_LAMBDA = 0.7   # 0 = max diversity, 1 = max relevance

# ── Generation prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a precise enterprise knowledge assistant. "
    "Answer the question using only the provided context passages. "
    "If the context contains information from multiple passages, synthesise across "
    "all of them rather than drawing from a single passage. "
    "If the context does not contain enough information to answer, "
    "say so explicitly rather than guessing."
)
