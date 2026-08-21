"""
Compute all Chapter 6 metrics from evaluation logs.

Requires (install before running):
    pip install ragas bert-score rouge-score datasets

Usage:
    python -m scripts.evaluation.compute_metrics \\
        --rag   rag/logs/eval_<timestamp>.jsonl \\
        --agent agent/logs/eval_<timestamp>.jsonl

Outputs:
    data/eval/results_<timestamp>.json   — all tables for Chapter 6
    Printed human-readable summary to stdout
"""

import argparse
import json
import statistics
import sys
import types

# langchain_community 0.4+ removed chat_models.vertexai (moved to langchain_google_vertexai).
# ragas 0.4.x still imports it at module level; stub it so the import succeeds.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    class _ChatVertexAI: pass
    _stub.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag import config as cfg

QUERIES_PATH  = ROOT / "data" / "eval" / "queries.jsonl"
CHUNKS_PATH   = cfg.INDEX_DIR / "unified_chunks.jsonl"
EVAL_DIR      = ROOT / "data" / "eval"

# Cost proxy: GPT-4o pricing (USD per 1M tokens)
COST_PER_M_INPUT  = 2.50
COST_PER_M_OUTPUT = 10.00

# Agent tool name → source_type for routing accuracy
TOOL_TO_SOURCE = {
    "search_github_issues":   "github_issue",
    "search_markdown_docs":   "markdown_doc",
    "search_pdf_whitepapers": "pdf_whitepaper",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_queries() -> dict[str, dict]:
    """Returns {question_text: query_record}."""
    qs = load_jsonl(QUERIES_PATH)
    return {q["question"]: q for q in qs}


def load_chunk_texts() -> dict[str, str]:
    """Returns {chunk_id: text}."""
    lookup = {}
    with open(CHUNKS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                lookup[c["chunk_id"]] = c["text"]
    return lookup


# ── RAGAS ────────────────────────────────────────────────────────────────────

def compute_ragas(records: list[dict], query_map: dict, chunk_texts: dict) -> list[dict]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics._answer_relevance import answer_relevancy
        from ragas.metrics._context_precision import context_precision
        from ragas.metrics._faithfulness import faithfulness
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings as LCHFEmb
    except ImportError as exc:
        print(f"  WARNING: RAGAS not available ({exc}). Skipping RAGAS metrics.", file=sys.stderr)
        return [{"context_precision": None, "faithfulness": None, "answer_relevancy": None}
                for _ in records]

    # Use the same LLM as the pipelines as judge
    judge_llm = LangchainLLMWrapper(ChatOpenAI(
        base_url=cfg.AI_REFINERY_BASE_URL,
        api_key=cfg.AI_REFINERY_API_KEY,
        model=cfg.LLM_MODEL,
        temperature=0.0,
        default_headers=cfg.LLM_DEFAULT_HEADERS,
    ))
    # Local BGE embeddings — AI Refinery has no embeddings endpoint
    judge_embeddings = LangchainEmbeddingsWrapper(LCHFEmb(
        model_name=cfg.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    ))

    for metric in (context_precision, faithfulness, answer_relevancy):
        metric.llm = judge_llm
    answer_relevancy.embeddings = judge_embeddings

    questions, answers, contexts, ground_truths = [], [], [], []
    for rec in records:
        q_rec = query_map.get(rec["query"])
        questions.append(rec["query"])
        answers.append(rec.get("answer") or "")
        ground_truths.append(q_rec["ground_truth"] if q_rec else "")
        ctx_texts = [
            chunk_texts.get(c["chunk_id"], c.get("title", ""))
            for c in rec.get("retrieved", [])
        ]
        contexts.append(ctx_texts if ctx_texts else [""])

    dataset = Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": ground_truths,
    })

    # Do NOT pass llm= or embeddings= to evaluate() — metrics already have them set above
    result = evaluate(dataset, metrics=[context_precision, faithfulness, answer_relevancy])
    df = result.to_pandas()

    return [
        {
            "context_precision": row.get("context_precision"),
            "faithfulness":      row.get("faithfulness"),
            "answer_relevancy":  row.get("answer_relevancy"),
        }
        for _, row in df.iterrows()
    ]


# ── BERTScore ────────────────────────────────────────────────────────────────

def compute_bertscore(records: list[dict], query_map: dict) -> list[float | None]:
    try:
        from bert_score import score as bert_score_fn
    except ImportError as exc:
        print(f"  WARNING: bert-score not available ({exc}). Skipping BERTScore.", file=sys.stderr)
        return [None] * len(records)

    cands = [rec.get("answer") or "" for rec in records]
    refs  = [
        (query_map.get(rec["query"]) or {}).get("ground_truth", "")
        for rec in records
    ]

    # DeBERTa-v2 only has a fast (Rust) tokenizer; Python 3.14 sys.maxsize overflows the
    # Rust tokenizer's C-int when enable_truncation(max_length=sys.maxsize) is called.
    # Monkeypatch sent_encode to cap max_length at 512 before the Rust call.
    import bert_score.utils as _bsu
    _orig_sent_encode = _bsu.sent_encode
    def _patched_sent_encode(tokenizer, a):
        return tokenizer.encode(a, add_special_tokens=True, truncation=True, max_length=512)
    _bsu.sent_encode = _patched_sent_encode
    try:
        _, _, F1 = bert_score_fn(
            cands, refs,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
            rescale_with_baseline=True,
            verbose=False,
            nthreads=1,
        )
    finally:
        _bsu.sent_encode = _orig_sent_encode
    return [float(f) for f in F1.tolist()]


# ── ROUGE-L ───────────────────────────────────────────────────────────────────

def compute_rouge(records: list[dict], query_map: dict) -> list[float]:
    try:
        from rouge_score import rouge_scorer as rs_mod
    except ImportError as exc:
        print(f"  WARNING: rouge-score not available ({exc}). Skipping ROUGE-L.", file=sys.stderr)
        return [None] * len(records)

    scorer = rs_mod.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for rec in records:
        cand = rec.get("answer") or ""
        ref  = (query_map.get(rec["query"]) or {}).get("ground_truth", "")
        scores.append(scorer.score(ref, cand)["rougeL"].fmeasure)
    return scores


# ── Routing accuracy (agent) ──────────────────────────────────────────────────

def compute_routing(records: list[dict], query_map: dict) -> dict:
    total = correct = 0
    by_type: dict[str, dict] = {}

    for rec in records:
        q_rec = query_map.get(rec["query"])
        if not q_rec:
            continue

        qt = q_rec.get("query_type", "unknown")
        expected_st = q_rec.get("source_type", "")

        if expected_st == "none":  # unanswerable — not applicable
            continue

        sources_used = set(rec.get("sources_used", []))
        # Map tool names → source types
        source_types_used = {TOOL_TO_SOURCE.get(t, t) for t in sources_used}

        if expected_st == "cross_source":
            expected_set = {s for s in q_rec.get("source_ids", [])
                            # source_ids contains chunk IDs; can't derive source types here
                            # fall back to checking both expected types appear in sources_used
                            }
            # For cross_source, check if ≥2 distinct source types were used
            correct_flag = len(source_types_used) >= 2
        else:
            correct_flag = expected_st in source_types_used

        by_type.setdefault(qt, {"total": 0, "correct": 0})
        by_type[qt]["total"]   += 1
        by_type[qt]["correct"] += int(correct_flag)
        total   += 1
        correct += int(correct_flag)

    overall = correct / total if total else None
    return {
        "overall_pct":   round(overall * 100, 1) if overall is not None else None,
        "correct":       correct,
        "total":         total,
        "by_query_type": {
            qt: {
                "pct": round(v["correct"] / v["total"] * 100, 1) if v["total"] else None,
                **v,
            }
            for qt, v in by_type.items()
        },
    }


# ── Latency + cost ────────────────────────────────────────────────────────────

def compute_latency_cost(records: list[dict], query_map: dict) -> dict:
    totals_s = [r["latency"]["total_s"] for r in records if "latency" in r]
    tok_in   = [r["tokens"]["input"]    for r in records if "tokens"  in r]
    tok_out  = [r["tokens"]["output"]   for r in records if "tokens"  in r]

    def stats(vals):
        if not vals:
            return {}
        return {
            "mean":   round(statistics.mean(vals),   3),
            "median": round(statistics.median(vals), 3),
            "stdev":  round(statistics.stdev(vals),  3) if len(vals) > 1 else 0.0,
            "min":    round(min(vals), 3),
            "max":    round(max(vals), 3),
        }

    # Per query_type breakdown
    by_type: dict[str, list] = {}
    for rec in records:
        q_rec = query_map.get(rec["query"])
        qt = q_rec.get("query_type", "unknown") if q_rec else "unknown"
        by_type.setdefault(qt, [])
        if "latency" in rec:
            by_type[qt].append(rec["latency"]["total_s"])

    total_input  = sum(tok_in)
    total_output = sum(tok_out)
    cost_usd = (total_input / 1_000_000 * COST_PER_M_INPUT +
                total_output / 1_000_000 * COST_PER_M_OUTPUT)

    return {
        "latency_s":          stats(totals_s),
        "latency_by_qtype":   {qt: stats(vs) for qt, vs in by_type.items()},
        "tokens_input_total":  total_input,
        "tokens_output_total": total_output,
        "tokens_total":        total_input + total_output,
        "cost_usd_estimate":   round(cost_usd, 4),
        "cost_note":           "Proxy pricing: GPT-4o $2.50/1M input, $10.00/1M output",
    }


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 4) if vals else None


def aggregate_quality(ragas_rows, bertscore, rouge, records, query_map):
    by_type: dict[str, dict] = {}
    for i, rec in enumerate(records):
        q_rec = query_map.get(rec["query"])
        qt = q_rec.get("query_type", "unknown") if q_rec else "unknown"
        entry = by_type.setdefault(qt, {"cp": [], "faith": [], "ar": [], "bs": [], "rl": []})
        r = ragas_rows[i] if i < len(ragas_rows) else {}
        entry["cp"].append(r.get("context_precision"))
        entry["faith"].append(r.get("faithfulness"))
        entry["ar"].append(r.get("answer_relevancy"))
        entry["bs"].append(bertscore[i] if i < len(bertscore) else None)
        entry["rl"].append(rouge[i] if i < len(rouge) else None)

    return {
        "overall": {
            "context_precision": _mean([r.get("context_precision") for r in ragas_rows]),
            "faithfulness":      _mean([r.get("faithfulness")      for r in ragas_rows]),
            "answer_relevancy":  _mean([r.get("answer_relevancy")  for r in ragas_rows]),
            "bertscore_f1":      _mean(bertscore),
            "rouge_l":           _mean(rouge),
        },
        "by_query_type": {
            qt: {
                "context_precision": _mean(v["cp"]),
                "faithfulness":      _mean(v["faith"]),
                "answer_relevancy":  _mean(v["ar"]),
                "bertscore_f1":      _mean(v["bs"]),
                "rouge_l":           _mean(v["rl"]),
            }
            for qt, v in by_type.items()
        },
    }


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_section(title: str, data: dict) -> None:
    print(f"\n{'─'*64}")
    print(f"  {title}")
    print(f"{'─'*64}")
    print(json.dumps(data, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def process_system(label: str, log_path: Path, query_map: dict, chunk_texts: dict) -> dict:
    print(f"\n{'='*64}")
    print(f"  Processing {label}  ({log_path.name})")
    print(f"{'='*64}")

    records = load_jsonl(log_path)
    print(f"  {len(records)} log records loaded.")

    # Cache RAGAS rows to disk so they survive a BERTScore crash (RAGAS is the expensive step)
    ragas_cache = EVAL_DIR / f"ragas_cache_{label.lower()}_{log_path.stem}.json"
    if ragas_cache.exists():
        with open(ragas_cache) as f:
            ragas_rows = json.load(f)
        print(f"  RAGAS scores loaded from cache ({ragas_cache.name})")
    else:
        print("  Computing RAGAS scores…")
        ragas_rows = compute_ragas(records, query_map, chunk_texts)
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with open(ragas_cache, "w") as f:
            json.dump(ragas_rows, f)
        print(f"  RAGAS scores cached → {ragas_cache.name}")

    print("  Computing BERTScore…")
    bs = compute_bertscore(records, query_map)

    print("  Computing ROUGE-L…")
    rl = compute_rouge(records, query_map)

    quality = aggregate_quality(ragas_rows, bs, rl, records, query_map)
    latency = compute_latency_cost(records, query_map)

    result = {"system": label, "n_queries": len(records), "quality": quality, "latency_cost": latency}

    if label.lower() == "agent":
        print("  Computing routing accuracy…")
        result["routing"] = compute_routing(records, query_map)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Chapter 6 evaluation metrics.")
    parser.add_argument("--rag",     required=True, help="Path to RAG eval log (JSONL)")
    parser.add_argument("--agent",   required=True, help="Path to agent eval log (JSONL)")
    parser.add_argument("--queries", default=str(QUERIES_PATH), help="Path to queries.jsonl")
    args = parser.parse_args()

    rag_path   = Path(args.rag)
    agent_path = Path(args.agent)
    q_path     = Path(args.queries)

    for p in (rag_path, agent_path, q_path):
        if not p.exists():
            print(f"Error: {p} not found.", file=sys.stderr)
            sys.exit(1)

    print("Loading queries and chunk texts…")
    query_map   = load_queries()
    chunk_texts = load_chunk_texts()
    print(f"  {len(query_map)} queries | {len(chunk_texts):,} chunks")

    rag_results   = process_system("RAG",   rag_path,   query_map, chunk_texts)
    agent_results = process_system("Agent", agent_path, query_map, chunk_texts)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_section("RAG — Quality",       rag_results["quality"])
    print_section("Agent — Quality",     agent_results["quality"])
    print_section("RAG — Latency/Cost",  rag_results["latency_cost"])
    print_section("Agent — Latency/Cost", agent_results["latency_cost"])
    if "routing" in agent_results:
        print_section("Agent — Routing Accuracy", agent_results["routing"])

    # ── Write results JSON ────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = EVAL_DIR / f"results_{ts}.json"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump({"rag": rag_results, "agent": agent_results}, f, indent=2)

    print(f"\nResults written → {out_path}")


if __name__ == "__main__":
    main()
