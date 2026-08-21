"""
Generate evaluation query candidates from the corpus.

Writes:
  data/eval/candidates.jsonl          — factual, synthesis, ambiguous candidates
  data/eval/unanswerable_template.jsonl — 15 blank stubs for manual authoring

Run from repo root:
    python -m scripts.evaluation.generate_queries

After running:
1. Review data/eval/candidates.jsonl — delete rejects, edit as needed.
2. Fill in data/eval/unanswerable_template.jsonl (15 questions absent from corpus).
3. Merge both files into data/eval/queries.jsonl (target: 90 records total).
4. Run python -m scripts.evaluation.check_queries to validate.
"""

import json
import random
import re
import sys
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rag import config as cfg

SEED = 42
EVAL_DIR = ROOT / "data" / "eval"
CHUNKS_PATH = cfg.INDEX_DIR / "unified_chunks.jsonl"

# How many LLM calls to make per type (more than target to allow for rejects)
N_FACTUAL_SAMPLE   = 50   # → keep ~30 after review (10 per source type)
N_SYNTHESIS_PAIRS  = 40   # → keep ~30 after review (≥10 cross-source)
N_CROSS_PAIRS      = 20   # of the 40 pairs, 20 are cross-source
N_AMBIGUOUS_SAMPLE = 25   # factual candidates to rewrite → keep ~15

# ── Prompt templates ──────────────────────────────────────────────────────────

FACTUAL_SYSTEM = "You are an expert technical question designer for evaluation benchmarks."

FACTUAL_USER = """\
Document excerpt ({source_type}):

{text}

Design a factual question based on the passage above. Requirements:
- The question must be answerable solely from the provided text.
- The question must NOT quote or paraphrase the answer verbatim.
- The question must be specific enough to have exactly one correct answer.
- The answer must be 1-3 sentences.

Return JSON only, no markdown fences, no extra text:
{{"question": "...", "ground_truth": "..."}}"""

SYNTHESIS_SYSTEM = "You are an expert technical question designer for evaluation benchmarks."

SYNTHESIS_USER = """\
Document excerpt 1 ({type1}):
{text1}

Document excerpt 2 ({type2}):
{text2}

Design a synthesis question that requires information from BOTH passages above. Requirements:
- The question cannot be fully answered from either passage alone — both are needed.
- The question must NOT quote or paraphrase the answer verbatim.
- The answer must be 2-4 sentences, drawing on both sources.

Return JSON only, no markdown fences, no extra text:
{{"question": "...", "ground_truth": "..."}}"""

AMBIGUOUS_SYSTEM = "You are an expert technical question designer for adversarial evaluation."

AMBIGUOUS_USER = """\
Original question: {question}

Rewrite it as a genuinely ambiguous query by removing the key identifying detail \
(the technology name, version number, or specific system being described).
The rewritten question must still sound plausible and be related to the domain.

Return JSON only, no markdown fences, no extra text:
{{"question": "...", "ambiguity_note": "what was removed and why it creates ambiguity"}}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def stratified_sample(chunks: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample n chunks, roughly equal counts per source_type."""
    by_type: dict[str, list] = {}
    for c in chunks:
        by_type.setdefault(c["source_type"], []).append(c)

    types = sorted(by_type)
    per_type, remainder = divmod(n, len(types))
    result = []
    for i, t in enumerate(types):
        take = per_type + (1 if i < remainder else 0)
        result.extend(rng.sample(by_type[t], min(take, len(by_type[t]))))
    rng.shuffle(result)
    return result


def make_pairs(
    chunks: list[dict], n_total: int, n_cross: int, rng: random.Random
) -> list[tuple[dict, dict, str]]:
    """Return (chunk1, chunk2, source_type_label) pairs."""
    by_type: dict[str, list] = {}
    for c in chunks:
        by_type.setdefault(c["source_type"], []).append(c)
    types = sorted(by_type)

    pairs: list[tuple] = []

    # Cross-source pairs
    for _ in range(n_cross):
        t1, t2 = rng.sample(types, 2)
        pairs.append((rng.choice(by_type[t1]), rng.choice(by_type[t2]), "cross_source"))

    # Within-source pairs (prefer different doc_ids for richer synthesis)
    for _ in range(n_total - n_cross):
        t = rng.choice(types)
        pool = by_type[t]
        if len(pool) < 2:
            continue
        c1 = rng.choice(pool)
        candidates = [c for c in pool if c["doc_id"] != c1["doc_id"]] or pool
        c2 = rng.choice(candidates)
        pairs.append((c1, c2, t))

    return pairs[:n_total]


def call_llm(client: OpenAI, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=cfg.LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.7,
        max_tokens=512,
    )
    return resp.choices[0].message.content or ""


def parse_json(text: str) -> dict:
    """Strip markdown fences then parse JSON."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(SEED)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        base_url=cfg.AI_REFINERY_BASE_URL,
        api_key=cfg.AI_REFINERY_API_KEY,
        default_headers=cfg.LLM_DEFAULT_HEADERS,
    )

    print("Loading chunks…")
    all_chunks = load_chunks()
    print(f"  {len(all_chunks):,} chunks loaded from {CHUNKS_PATH.name}")

    factual: list[dict] = []
    synthesis: list[dict] = []
    ambiguous: list[dict] = []

    # ── Factual ───────────────────────────────────────────────────────────────
    sample = stratified_sample(all_chunks, N_FACTUAL_SAMPLE, rng)
    print(f"\nGenerating factual candidates ({len(sample)} LLM calls)…")

    for chunk in tqdm(sample, desc="factual"):
        user_msg = FACTUAL_USER.format(
            source_type=chunk["source_type"],
            text=chunk["text"][:2000],
        )
        try:
            parsed = parse_json(call_llm(client, FACTUAL_SYSTEM, user_msg))
            factual.append({
                "question":    parsed["question"],
                "ground_truth": parsed["ground_truth"],
                "query_type":  "factual",
                "source_type": chunk["source_type"],
                "source_ids":  [chunk["chunk_id"]],
                "notes":       "",
            })
        except Exception as exc:
            tqdm.write(f"  skip {chunk['chunk_id']}: {exc}")

    print(f"  {len(factual)} factual candidates accepted.")

    # ── Synthesis ─────────────────────────────────────────────────────────────
    pairs = make_pairs(all_chunks, N_SYNTHESIS_PAIRS, N_CROSS_PAIRS, rng)
    print(f"\nGenerating synthesis candidates ({len(pairs)} LLM calls)…")

    for c1, c2, st in tqdm(pairs, desc="synthesis"):
        user_msg = SYNTHESIS_USER.format(
            type1=c1["source_type"], text1=c1["text"][:1200],
            type2=c2["source_type"], text2=c2["text"][:1200],
        )
        try:
            parsed = parse_json(call_llm(client, SYNTHESIS_SYSTEM, user_msg))
            synthesis.append({
                "question":    parsed["question"],
                "ground_truth": parsed["ground_truth"],
                "query_type":  "synthesis",
                "source_type": st,
                "source_ids":  [c1["chunk_id"], c2["chunk_id"]],
                "notes":       "",
            })
        except Exception as exc:
            tqdm.write(f"  skip pair ({c1['chunk_id']}, {c2['chunk_id']}): {exc}")

    print(f"  {len(synthesis)} synthesis candidates accepted.")

    # ── Ambiguous ─────────────────────────────────────────────────────────────
    to_rewrite = rng.sample(factual, min(N_AMBIGUOUS_SAMPLE, len(factual)))
    print(f"\nGenerating ambiguous candidates ({len(to_rewrite)} LLM calls)…")

    for orig in tqdm(to_rewrite, desc="ambiguous"):
        user_msg = AMBIGUOUS_USER.format(question=orig["question"])
        try:
            parsed = parse_json(call_llm(client, AMBIGUOUS_SYSTEM, user_msg))
            ambiguous.append({
                "question":    parsed["question"],
                "ground_truth": "",
                "query_type":  "ambiguous",
                "source_type": orig["source_type"],
                "source_ids":  orig["source_ids"],
                "notes":       parsed.get("ambiguity_note", ""),
            })
        except Exception as exc:
            tqdm.write(f"  skip ambiguous rewrite: {exc}")

    print(f"  {len(ambiguous)} ambiguous candidates accepted.")

    # ── Assign IDs and write candidates.jsonl ─────────────────────────────────
    all_candidates: list[dict] = []
    for i, rec in enumerate(factual,   1): all_candidates.append({"id": f"q_f{i:03d}", **rec})
    for i, rec in enumerate(synthesis, 1): all_candidates.append({"id": f"q_s{i:03d}", **rec})
    for i, rec in enumerate(ambiguous, 1): all_candidates.append({"id": f"q_a{i:03d}", **rec})

    candidates_path = EVAL_DIR / "candidates.jsonl"
    with open(candidates_path, "w") as f:
        for rec in all_candidates:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(all_candidates)} candidates → {candidates_path}")

    # ── Write unanswerable template ───────────────────────────────────────────
    template_path = EVAL_DIR / "unanswerable_template.jsonl"
    with open(template_path, "w") as f:
        for i in range(1, 16):
            f.write(json.dumps({
                "id":           f"q_u{i:03d}",
                "question":     "TODO: question about a topic absent from the corpus",
                "ground_truth": "The corpus does not contain information about this topic.",
                "query_type":   "unanswerable",
                "source_type":  "none",
                "source_ids":   [],
                "notes":        "TODO: verify no chunk answers this via retrieval check",
            }) + "\n")

    print(f"Wrote 15 unanswerable stubs → {template_path}")
    print()
    print("Next steps:")
    print("  1. Review candidates.jsonl — delete rejects, edit accepted records")
    print("  2. Fill in unanswerable_template.jsonl (15 questions not in corpus)")
    print("  3. Merge both into data/eval/queries.jsonl (target: 90 records)")
    print("  4. python -m scripts.evaluation.check_queries")


if __name__ == "__main__":
    main()
