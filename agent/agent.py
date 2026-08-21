"""
ReAct agent construction via LangGraph.

LangChain 1.x moved create_react_agent to langgraph.prebuilt. It uses the
model's native tool-calling API rather than text-parsed Thought/Action strings.
The ReAct reasoning pattern (Yao et al.) is preserved: the model reasons
internally and emits tool calls, receives observations, and iterates until it
produces a final answer — the structural loop is identical, expressed via
function-calling rather than free-text parsing.

build_agent() returns both the compiled StateGraph and the ToolBundle so
AgentPipeline can access retriever call histories after invoke() completes.
"""

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from .tools import ToolBundle
from . import config


def build_agent() -> tuple:
    """
    Returns (app, bundle).
    - app   : compiled LangGraph StateGraph runnable
    - bundle: ToolBundle instance whose retrievers hold per-call chunk history
    """
    llm = ChatOpenAI(
        base_url=config.AI_REFINERY_BASE_URL,
        api_key=config.AI_REFINERY_API_KEY,
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        default_headers=config.LLM_DEFAULT_HEADERS,
    )

    bundle = ToolBundle()
    tools  = bundle.get_tools()

    app = create_react_agent(
        llm,
        tools,
        prompt=config.SYSTEM_PROMPT,
    )
    return app, bundle
