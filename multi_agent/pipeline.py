"""
Multi-agent pipeline: query → parallel retrieval → synthesis → log.

Log schema is a strict superset of agent/pipeline.py:
  - All shared fields (timestamp, system, query, answer, finish_reason,
    retrieved, latency, tokens) are structurally identical.
  - agent_trace has one entry per source (step = source position, tool = source
    name, tool_input = query, observation = formatted chunk text).
  - tool_calls = 3 (always all sources queried).
  - sources_used = sources that returned at least one chunk.
  - system = "multi_agent"

Usage:
    from multi_agent.pipeline import MultiAgentPipeline
    pipeline = MultiAgentPipeline()
    result = pipeline.query("What does the CNCF security whitepaper say about supply chains?")
    print(result["answer"])
"""

import json
import time
from datetime import datetime, timezone

try:
    from langchain_community.callbacks import get_openai_callback
    _HAS_CALLBACK = True
except ImportError:
    _HAS_CALLBACK = False

from .graph import build_graph
from . import config


class MultiAgentPipeline:
    def __init__(self):
        self.app = build_graph()
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._log_path = (
            config.LOG_DIR / f"multi_agent_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.jsonl"
        )

    def query(self, question: str) -> dict:
        t_start = time.perf_counter()

        input_tokens  = 0
        output_tokens = 0
        finish_reason = "stop"
        source_results: list[dict] = []
        answer = ""

        invoke_input = {
            "query":          question,
            "source_results": [],
            "answer":         "",
            "finish_reason":  "stop",
        }

        cb = None
        try:
            if _HAS_CALLBACK:
                with get_openai_callback() as cb:
                    result = self.app.invoke(invoke_input)
                input_tokens  = cb.prompt_tokens
                output_tokens = cb.completion_tokens
            else:
                result = self.app.invoke(invoke_input)

            answer         = result.get("answer", "")
            finish_reason  = result.get("finish_reason", "stop")
            source_results = result.get("source_results", [])

        except Exception as exc:
            finish_reason = "error"
            if cb is not None:
                input_tokens  = cb.prompt_tokens
                output_tokens = cb.completion_tokens
            # Log the error string as the answer so the record is not silently empty.
            answer = f"[pipeline error: {exc}]"

        t_total = time.perf_counter() - t_start

        # ── Build agent_trace (one entry per source, in config.ALL_SOURCES order) ─
        # Sort source_results to match ALL_SOURCES order for determinism.
        ordered = {r["source"]: r for r in source_results}
        agent_trace = []
        retrieved_flat: list[dict] = []
        for step, src in enumerate(config.ALL_SOURCES, 1):
            res = ordered.get(src, {"source": src, "chunks": [], "observation": ""})
            agent_trace.append({
                "step":        step,
                "tool":        src,
                "tool_input":  question,
                "observation": res["observation"][:3000],
            })
            for chunk in res.get("chunks", []):
                retrieved_flat.append({
                    "step":         step,
                    "tool":         src,
                    "chunk_id":     chunk.get("chunk_id", ""),
                    "source_type":  chunk.get("source_type", src),
                    "doc_id":       chunk.get("doc_id", ""),
                    "title":        chunk.get("title", ""),
                    "text":         chunk.get("text", ""),
                    "faiss_score":  chunk.get("faiss_score"),
                    "rerank_score": chunk.get("rerank_score"),
                })

        sources_used = sorted(
            src for src in config.ALL_SOURCES
            if ordered.get(src, {}).get("chunks")
        )

        result_dict = {
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "system":        "multi_agent",
            "query":         question,
            "answer":        answer,
            "finish_reason": finish_reason,
            "retrieved":     retrieved_flat,
            "latency": {
                "total_s": round(t_total, 4),
            },
            "tokens": {
                "input":  input_tokens,
                "output": output_tokens,
                "total":  input_tokens + output_tokens,
            },
            "agent_trace":  agent_trace,
            "tool_calls":   len(config.ALL_SOURCES),
            "sources_used": sources_used,
        }

        self._log(result_dict)
        return result_dict

    def _log(self, result: dict):
        with open(self._log_path, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    pipeline = MultiAgentPipeline()
    q = "What does the CNCF security whitepaper recommend about supply chain security?"
    print(f"Query: {q}\n")
    result = pipeline.query(q)
    print(f"Answer: {result['answer']}\n")
    print(f"Latency: {result['latency']['total_s']:.2f}s")
    print(f"Tokens:  {result['tokens']['total']}")
    print(f"Sources: {result['sources_used']}")
