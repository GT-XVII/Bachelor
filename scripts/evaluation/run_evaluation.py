"""
Run both pipelines over all evaluation queries.

Usage:
    python -m scripts.evaluation.run_evaluation
    python -m scripts.evaluation.run_evaluation --system rag
    python -m scripts.evaluation.run_evaluation --system agent
    python -m scripts.evaluation.run_evaluation --queries data/eval/queries.jsonl

Writes:
    rag/logs/eval_<timestamp>.jsonl
    agent/logs/eval_<timestamp>.jsonl

Estimated runtime: ~30-35 minutes for both systems over 90 queries on VPN.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_QUERIES = ROOT / "data" / "eval" / "queries.jsonl"


def load_queries(path: Path) -> list[dict]:
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def run_pipeline(pipeline, queries: list[dict], label: str) -> None:
    log_path = pipeline._log_path
    n = len(queries)
    print(f"\n{'='*64}")
    print(f"  {label}  |  {n} queries  |  log → {log_path.name}")
    print(f"{'='*64}")

    for i, q in enumerate(queries, 1):
        preview = q["question"][:70] + ("…" if len(q["question"]) > 70 else "")
        print(f"[{i:3d}/{n}] {q.get('query_type','?'):12s} | {preview}", flush=True)
        try:
            result = pipeline.query(q["question"])
            lat = result["latency"]["total_s"]
            tok_in  = result["tokens"]["input"]
            tok_out = result["tokens"]["output"]
            print(f"          → {lat:.1f}s | in {tok_in} / out {tok_out} tokens")
        except Exception as exc:
            print(f"          ERROR: {exc}", file=sys.stderr)

    print(f"\n{label} complete. {n} results written to {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation pipelines.")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES),
                        help="Path to queries.jsonl (default: data/eval/queries.jsonl)")
    parser.add_argument("--system", choices=["rag", "agent", "both"], default="both",
                        help="Which pipeline to run (default: both)")
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"Error: {queries_path} not found.", file=sys.stderr)
        print("Run generate_queries.py and complete manual review first.", file=sys.stderr)
        sys.exit(1)

    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} queries from {queries_path}")

    # Shared timestamp so RAG and agent logs are paired by filename
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    if args.system in ("rag", "both"):
        from rag.pipeline import RAGPipeline
        rag = RAGPipeline()
        rag._log_path = rag._log_path.parent / f"eval_{ts}.jsonl"
        run_pipeline(rag, queries, "RAG")

    if args.system in ("agent", "both"):
        from agent.pipeline import AgentPipeline
        agent = AgentPipeline()
        agent._log_path = agent._log_path.parent / f"eval_{ts}.jsonl"
        run_pipeline(agent, queries, "Agent")

    print("\nEvaluation run complete.")


if __name__ == "__main__":
    main()
