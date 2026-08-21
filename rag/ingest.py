"""
Corpus ingestion: load JSONL → chunk → embed → build FAISS index.

Run once (or whenever the corpus changes):
    python -m rag.ingest
"""

import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer

from . import config


def load_corpus() -> list[dict]:
    records = []
    for source_type, path in config.CORPUS_FILES.items():
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping.")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    print(f"Loaded {len(records)} records from corpus.")
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
    """Expand each record into one or more chunk dicts."""
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
    print(f"Built {len(chunks)} chunks from {len(records)} records.")
    return chunks


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {config.EMBEDDING_MODEL}...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # required for cosine via IndexFlatIP
        convert_to_numpy=True,
    )
    print(f"Embedding done in {time.time()-t0:.1f}s — shape {embeddings.shape}")
    return embeddings.astype("float32")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """IndexFlatIP on L2-normalised vectors = cosine similarity, exact search."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save(index: faiss.IndexFlatIP, chunks: list[dict]):
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))
    with open(config.CHUNKS_PATH, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"Saved index  → {config.INDEX_PATH}")
    print(f"Saved chunks → {config.CHUNKS_PATH}")


def main():
    records    = load_corpus()
    chunks     = build_chunks(records)
    model      = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = embed_chunks(chunks, model)
    index      = build_index(embeddings)
    save(index, chunks)


if __name__ == "__main__":
    main()
