"""
Per-source corpus ingestion: load JSONL → chunk → embed → build FAISS indexes.

One index is built per source type, mirroring the enterprise-silo architecture
described in Chapter 3.3. All parameters are identical to rag/ingest.py so
that retrieval quality per tool call is directly comparable to the RAG pipeline.

Run once (or whenever the corpus changes):
    python -m agent.ingest
"""

import json
import random
import time
import numpy as np
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer

from . import config


def load_source(source_type: str) -> list[dict]:
    path = config.CORPUS_FILES[source_type]
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping.")
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    limit = config.SOURCE_LIMITS.get(source_type)
    if limit and len(records) > limit:
        random.seed(42)
        records = random.sample(records, limit)
        print(f"  Downsampled {source_type} to {limit} records.")
    print(f"  Loaded {len(records)} {source_type} records.")
    return records


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-count windows."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks(records: list[dict]) -> list[dict]:
    chunks = []
    for rec in records:
        content = rec.get("content", "").strip()
        if not content:
            continue
        texts = chunk_text(content, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
        for i, text in enumerate(texts):
            chunks.append({
                "chunk_id":    f"{rec['id']}__chunk{i}",
                "source_type": rec["source_type"],
                "doc_id":      rec["id"],
                "title":       rec.get("title", ""),
                "text":        text,
                "metadata":    rec.get("metadata", {}),
            })
    print(f"  Built {len(chunks)} chunks from {len(records)} records.")
    return chunks


def embed_and_index(chunks: list[dict], model: SentenceTransformer) -> faiss.IndexFlatIP:
    texts = [c["text"] for c in chunks]
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"  Embedding done in {time.time() - t0:.1f}s — shape {embeddings.shape}")
    embeddings = embeddings.astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_source(source_type: str, index: faiss.IndexFlatIP, chunks: list[dict]):
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATHS[source_type]))
    with open(config.CHUNKS_PATHS[source_type], "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"  Saved → {config.INDEX_PATHS[source_type].name}  ({index.ntotal} vectors)")


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    for source_type in ("github_issue", "markdown_doc", "pdf_whitepaper"):
        print(f"\n── {source_type} ──")
        records = load_source(source_type)
        if not records:
            continue
        chunks = build_chunks(records)
        index  = embed_and_index(chunks, model)
        save_source(source_type, index, chunks)
    print("\nDone.")


if __name__ == "__main__":
    main()
