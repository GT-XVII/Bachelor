"""LLM generation via Accenture AI Refinery (OpenAI-compatible API)."""

from openai import OpenAI
from . import config


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c.get("metadata", {}).get("url") or c.get("doc_id", "")
        parts.append(f"[{i}] ({c['source_type']}) {c['title']}\nSource: {source}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _build_messages(query: str, chunks: list[dict]) -> list[dict]:
    context = _build_context(chunks)
    user_content = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above. "
        "Cite the passage number(s) in square brackets where relevant."
    )
    return [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


class Generator:
    def __init__(self):
        self.client = OpenAI(
            base_url=config.AI_REFINERY_BASE_URL,
            api_key=config.AI_REFINERY_API_KEY,
            default_headers=config.LLM_DEFAULT_HEADERS,
        )

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """
        Returns:
            answer       str   — the model's response text
            input_tokens  int  — prompt token count
            output_tokens int  — completion token count
        """
        messages = _build_messages(query, chunks)

        response = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            extra_body=config.LLM_EXTRA_BODY,   # None for AI Refinery, per integration report
        )

        choice = response.choices[0]
        usage  = response.usage

        return {
            "answer":        choice.message.content,
            "finish_reason": choice.finish_reason,
            "input_tokens":  usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }
