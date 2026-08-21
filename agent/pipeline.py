"""
Full agent pipeline: query → ReAct loop → log.

Log schema is a strict superset of rag/pipeline.py: shared outer fields
(timestamp, system, query, answer, finish_reason, latency, tokens, retrieved)
are structurally identical; agent-specific fields (agent_trace, tool_calls,
sources_used) are appended.

Usage:
    from agent.pipeline import AgentPipeline
    pipeline = AgentPipeline()
    result = pipeline.query("What does the CNCF security whitepaper say about supply chains?")
    print(result["answer"])
"""

import json
import time
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

try:
    from langchain_community.callbacks import get_openai_callback
    _HAS_CALLBACK = True
except ImportError:
    _HAS_CALLBACK = False

from .agent import build_agent
from . import config

# Each tool call is ~2 LangGraph supersteps (agent node + tools node).
# Allow MAX_ITERATIONS full tool-call cycles plus a final answer step.
_RECURSION_LIMIT = config.MAX_ITERATIONS * 2 + 1


class AgentPipeline:
    def __init__(self):
        self.app, self.bundle = build_agent()
        self._valid_tools = {t.name for t in self.bundle.get_tools()}
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._log_path = (
            config.LOG_DIR / f"agent_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.jsonl"
        )

    def query(self, question: str) -> dict:
        t_start = time.perf_counter()

        input_tokens  = 0
        output_tokens = 0
        messages      = []
        finish_reason = "stop"

        invoke_input  = {"messages": [("human", question)]}
        invoke_config = {"recursion_limit": _RECURSION_LIMIT}

        cb = None
        try:
            if _HAS_CALLBACK:
                with get_openai_callback() as cb:
                    result = self.app.invoke(invoke_input, config=invoke_config)
                input_tokens  = cb.prompt_tokens
                output_tokens = cb.completion_tokens
            else:
                result = self.app.invoke(invoke_input, config=invoke_config)
            messages = result.get("messages", [])
        except GraphRecursionError:
            finish_reason = "max_iterations"
            if cb is not None:
                input_tokens  = cb.prompt_tokens
                output_tokens = cb.completion_tokens

        t_total = time.perf_counter() - t_start

        # ── Extract final answer ───────────────────────────────────────────────
        final_answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                content = msg.content
                if isinstance(content, list):
                    # Extract only text-type blocks; reasoning/thinking blocks are skipped.
                    content = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                    )
                final_answer = (content if isinstance(content, str) else str(content)).strip()
                break

        # Fallback: if all content blocks were thinking-type the primary loop yields "".
        # Try any text content from the last AIMessage regardless of block type.
        if not final_answer:
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    content = msg.content
                    if isinstance(content, list):
                        parts = [
                            b.get("text") or b.get("thinking", "")
                            for b in content if isinstance(b, dict)
                        ]
                        content = " ".join(p for p in parts if p)
                    if isinstance(content, str):
                        final_answer = content.strip()
                    if final_answer:
                        break

        # ── Build agent trace from message sequence ────────────────────────────
        # Pair each AIMessage tool_call with its ToolMessage response by id.
        agent_trace: list[dict] = []
        pending: dict[str, dict] = {}   # tool_call_id → partial trace entry

        step = 0
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    step += 1
                    pending[tc["id"]] = {
                        "step":       step,
                        "tool":       tc["name"],
                        "tool_input": tc["args"].get("query", str(tc["args"])),
                    }
            elif isinstance(msg, ToolMessage) and msg.tool_call_id in pending:
                entry = pending.pop(msg.tool_call_id)
                entry["observation"] = (
                    msg.content[:3000] if isinstance(msg.content, str)
                    else str(msg.content)[:3000]
                )
                agent_trace.append(entry)

        # ── Collect structured chunk metadata from retriever histories ─────────
        # Reconstruct steps as (action-like, observation) tuples for ToolBundle
        pseudo_steps = [
            (_ToolCallProxy(e["tool"], e["tool_input"]), e.get("observation", ""))
            for e in agent_trace
        ]
        retrieved = self.bundle.collect_retrieved(pseudo_steps)

        # Exclude hallucinated tool names from evaluation fields; full trace is
        # preserved in agent_trace for diagnostics.
        valid_trace  = [e for e in agent_trace if e["tool"] in self._valid_tools]
        sources_used = sorted({e["tool"] for e in valid_trace})

        result_dict = {
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "system":       "agent",
            "query":        question,
            "answer":       final_answer,
            "finish_reason": finish_reason,
            # retrieved context — structured chunk dicts (superset of RAG schema)
            "retrieved":    retrieved,
            # latency — full ReAct loop wall-clock time
            "latency": {
                "total_s": round(t_total, 4),
            },
            # token usage via OpenAI callback; 0 if langchain_community unavailable
            "tokens": {
                "input":  input_tokens,
                "output": output_tokens,
                "total":  input_tokens + output_tokens,
            },
            # agent-specific fields
            "agent_trace":  agent_trace,
            "tool_calls":   len(valid_trace),
            "sources_used": sources_used,
        }

        self._log(result_dict)
        return result_dict

    def _log(self, result: dict):
        with open(self._log_path, "a") as f:
            f.write(json.dumps(result) + "\n")


class _ToolCallProxy:
    """Minimal stand-in for LangChain AgentAction, used by ToolBundle.collect_retrieved."""
    def __init__(self, tool: str, tool_input: str):
        self.tool       = tool
        self.tool_input = tool_input
