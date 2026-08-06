import logging
import os
from typing import Any, Optional
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from deepagents import CompiledSubAgent, create_deep_agent

logger = logging.getLogger()


# =============================================================================
# 1. Middleware: Tool error handling
# =============================================================================

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Turn a tool exception into a ToolMessage the model can react to, instead of
    crashing the whole run - lets the agent retry with different arguments."""
    try:
        return handler(request)
    except Exception as exc:  # noqa: BLE001
        return ToolMessage(
            content=f"Tool '{request.tool_call['name']}' failed: {exc}. Try different arguments or another approach.",
            tool_call_id=request.tool_call["id"],
        )


# =============================================================================
# 2. Agent Factory
# =============================================================================

def build_agent(
    tools: Optional[list] = None,
    model_name: str = "gpt-4o",
    temperature: float = 1.0,
    system_prompt: str = (
        "You are a capable, tool-using assistant. Call tools whenever they would "
        "give a more accurate, current, or verifiable answer than reasoning alone. "
        "Think step by step, use tools as needed, then give a direct final answer."
    ),
    checkpointer: Optional[Any] = None,
    middlewares:Optional[Any]=None
):
    """Assemble the production agent powered exclusively by ChatOpenAI."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY environment variable is missing.")

    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )
    if middlewares :
        middlewares.append[handle_tool_errors]
    else:
        middlewares = [handle_tool_errors]
    return create_deep_agent(
        model=model,
        tools=tools or [],
        system_prompt=system_prompt,
        middleware=middlewares,
        checkpointer=checkpointer,
    )



def make_subagent(
    *,
    name: str,
    description: str,
    agent:CompiledStateGraph
):
    return CompiledSubAgent(
        name=name,
        description=description,
        runnable=agent,
    )
# =============================================================================
# 3. Demo REPL
# =============================================================================

def _demo() -> None:
    import dotenv
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    agent = build_agent()
    config = {"configurable": {"thread_id": "demo-thread"}}

    print("OpenAI agent ready. Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
        print("agent>", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    _demo()
