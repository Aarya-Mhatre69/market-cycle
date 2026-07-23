import ast
import itertools
import logging
import operator
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import ToolMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger("rotating_agent")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_cerebras import ChatCerebras
from langchain_openai import ChatOpenAI

# Keep each provider's own internal retry loop short - we WANT rate limits and
# transient errors to surface quickly so the rotation middleware below can jump
# to a different model instead of burning seconds retrying the same one.
_MODEL_DEFAULTS: dict[str, Any] = dict(temperature=0.6, max_retries=1, timeout=60)


# =============================================================================
# 1. Rotating model pool
# =============================================================================

@dataclass
class ModelSlot:
    """One entry in the rotation: a concrete chat model instance plus its health."""

    name: str  # human-readable id, e.g. "gemini-2.5-flash#0"
    model: Any  # a BaseChatModel instance (ChatOpenAI, ChatMistralAI, ...)
    weight: int = 1  # relative selection weight among currently-healthy slots
    cooldown_until: float = 0.0  # monotonic() timestamp; skip slot until then
    consecutive_failures: int = 0

    @property
    def is_available(self) -> bool:
        return time.monotonic() >= self.cooldown_until


class RotatingModelPool:
    """
    Thread-safe round-robin + weighted pool of chat models.

    - Spreads load: rotates to a different model/provider/key on every call rather
      than hammering one endpoint, so you stay further from any single rate limit.
    - Self-heals: when a slot throws a rate-limit or transient error, it's put on
      an exponential-backoff cooldown and skipped until it expires; everything else
      keeps serving.
    - Degrades gracefully: if every slot is on cooldown simultaneously, it waits
      (capped) for the soonest one to free up rather than failing outright.
    """

    def __init__(self, slots: list[ModelSlot], base_cooldown: float = 20.0, max_cooldown: float = 300.0):
        if not slots:
            raise ValueError("RotatingModelPool needs at least one model slot")
        self._slots = slots
        self._lock = threading.Lock()
        self._cycle = itertools.cycle(range(len(slots)))
        self._base_cooldown = base_cooldown
        self._max_cooldown = max_cooldown

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def primary(self) -> Any:
        """The first configured model - used only as create_agent's static default;
        every real turn is routed through the rotation middleware instead."""
        return self._slots[0].model

    def next_available(self, exclude: Optional[set[str]] = None, wait_cap: float = 30.0) -> ModelSlot:
        exclude = exclude or set()
        deadline = time.monotonic() + wait_cap
        while True:
            with self._lock:
                order = [next(self._cycle) for _ in range(len(self._slots))]
                candidates = [
                    self._slots[i]
                    for i in order
                    if self._slots[i].is_available and self._slots[i].name not in exclude
                ]
                if candidates:
                    return random.choices(candidates, weights=[c.weight for c in candidates], k=1)[0]
                soonest = min(self._slots, key=lambda s: s.cooldown_until)
            if time.monotonic() >= deadline:
                # Every slot is either excluded or cooling down and we've waited
                # long enough - hand back the one that frees up soonest anyway.
                # The middleware's own retry loop will absorb the difference.
                return soonest
            time.sleep(min(0.5, max(0.0, soonest.cooldown_until - time.monotonic())))

    def mark_failure(self, slot: ModelSlot, *, rate_limited: bool) -> None:
        with self._lock:
            slot.consecutive_failures += 1
            if rate_limited:
                cooldown = min(self._max_cooldown, self._base_cooldown * (2 ** (slot.consecutive_failures - 1)))
                cooldown += cooldown * 0.2 * random.random()  # jitter
                slot.cooldown_until = time.monotonic() + cooldown
                logger.warning("%s rate-limited - benched for %.1fs", slot.name, cooldown)

    def mark_success(self, slot: ModelSlot) -> None:
        with self._lock:
            slot.consecutive_failures = 0
            slot.cooldown_until = 0.0


# --------------------------------------------------------------------------- #
# Error classification - providers raise different exception types, so match
# defensively on status codes first, then on message content.
# --------------------------------------------------------------------------- #
_RATE_LIMIT_MARKERS = (
    "rate limit", "ratelimit", "429", "resourceexhausted", "resource_exhausted",
    "too many requests", "quota exceeded", "throttl",
)
_TRANSIENT_MARKERS = (
    "timeout", "timed out", "connection", "temporarily unavailable", "503",
    "502", "overloaded", "server error", "internal error",
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def _is_transient_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


# =============================================================================
# 2. Middleware: dynamic model rotation on every agent turn
# =============================================================================

def make_rotating_model_middleware(pool: RotatingModelPool, max_attempts: Optional[int] = None):
    """
    Build a `@wrap_model_call` middleware bound to `pool`. On every model turn the
    agent takes, it grabs the next healthy model from the pool; on rate-limit or
    transient failure it benches that slot and retries the *same* turn on the next
    one, up to `max_attempts` - all inside a single agent step.
    """
    attempts_cap = max_attempts or max(3, len(pool) + 1)

    @wrap_model_call
    def rotating_model_call(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        tried: set[str] = set()
        last_exc: Optional[BaseException] = None

        for attempt in range(1, attempts_cap + 1):
            slot = pool.next_available(exclude=tried)
            tried.add(slot.name)
            try:
                response = handler(request.override(model=slot.model))
            except Exception as exc:  # noqa: BLE001 - deliberately broad; classified below
                rate_limited = _is_rate_limit_error(exc)
                if not (rate_limited or _is_transient_error(exc)):
                    raise  # a real error (bad request, auth, etc.) - don't mask it
                pool.mark_failure(slot, rate_limited=rate_limited)
                last_exc = exc
                logger.warning("attempt %d/%d via %s failed (%s) - rotating", attempt, attempts_cap, slot.name, exc)
                continue
            else:
                pool.mark_success(slot)
                return response

        raise RuntimeError(f"All models exhausted after {attempts_cap} attempts") from last_exc

    return rotating_model_call


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
# 3. Default model pool, built from whatever credentials are in the environment
# =============================================================================

def _split_keys(env_var: str) -> list[str]:
    return [k.strip() for k in os.environ.get(env_var, "").split(",") if k.strip()]


def build_default_pool() -> RotatingModelPool:
    """
    Reads credentials from the environment and builds a rotation pool spanning
    every provider that's configured. Multiple comma-separated keys per provider
    (e.g. `OPENAI_API_KEYS="sk-a,sk-b"`) each become their own slot, so you can
    rotate within one provider as well as across providers.
    """
    slots: list[ModelSlot] = []

    if ChatGoogleGenerativeAI is not None:
        for i, key in enumerate(_split_keys("GOOGLE_API_KEYS") or _split_keys("GOOGLE_API_KEY")):
            slots.append(ModelSlot(
                name=f"gemini-2.5-flash#{i}",
                model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=key, **_MODEL_DEFAULTS),
            ))

    if ChatMistralAI is not None:
        for i, key in enumerate(_split_keys("MISTRAL_API_KEYS") or _split_keys("MISTRAL_API_KEY")):
            slots.append(ModelSlot(
                name=f"mistral-large#{i}",
                model=ChatMistralAI(model="mistral-large-latest", api_key=key, **_MODEL_DEFAULTS),
            ))

    if ChatCerebras is not None:
        for i, key in enumerate(_split_keys("CEREBRAS_API_KEYS") or _split_keys("CEREBRAS_API_KEY")):
            slots.append(ModelSlot(
                name=f"cerebras-llama#{i}",
                model=ChatCerebras(model="llama-3.3-70b", api_key=key, **_MODEL_DEFAULTS),
            ))

    if ChatOpenAI is not None:
        for i, key in enumerate(_split_keys("OPENAI_API_KEYS") or _split_keys("OPENAI_API_KEY")):
            slots.append(ModelSlot(
                name=f"openai-gpt#{i}",
                model=ChatOpenAI(model="gpt-5.4-mini", api_key=key, **_MODEL_DEFAULTS),
            ))

        # Z.AI / GLM (or any other OpenAI-compatible endpoint) - same client class,
        # just a different base_url. Swap ZAI_BASE_URL to point this at Together,
        # vLLM, OpenRouter's OpenAI-compat mode, etc.
        for i, key in enumerate(_split_keys("ZAI_API_KEYS") or _split_keys("ZAI_API_KEY")):
            slots.append(ModelSlot(
                name=f"zai-glm#{i}",
                model=ChatOpenAI(
                    model=os.environ.get("ZAI_MODEL", "glm-5.2"),
                    api_key=key,
                    base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/"),
                    **_MODEL_DEFAULTS,
                ),
            ))

    if not slots:
        raise RuntimeError(
            "No model credentials found. Set at least one of: "
            "GOOGLE_API_KEY(S), MISTRAL_API_KEY(S), CEREBRAS_API_KEY(S), "
            "OPENAI_API_KEY(S), ZAI_API_KEY(S)."
        )

    logger.info("model pool ready with %d slot(s): %s", len(slots), [s.name for s in slots])
    return RotatingModelPool(slots)


# =============================================================================
# 4. Example tools - replace/extend with anything: @tool functions or BaseTool
#    instances all work identically with create_agent.
# =============================================================================

@tool
def web_search_stub(query: str) -> str:
    """Placeholder search tool. Swap in a real web/search API call here."""
    return f"[stub] would search the web for: {query}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4) / 5'."""
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported expression: {expression!r}")

    return str(_eval(ast.parse(expression, mode="eval").body))


DEFAULT_TOOLS = [web_search_stub, calculator]


# =============================================================================
# 5. Agent factory
# =============================================================================

def build_agent(
    tools: Optional[list] = None,
    pool: Optional[RotatingModelPool] = None,
    system_prompt: str = (
        "You are a capable, tool-using assistant. Call tools whenever they would "
        "give a more accurate, current, or verifiable answer than reasoning alone. "
        "Think step by step, use tools as needed, then give a direct final answer."
    ),
    checkpointer: Optional[Any] = None,
    max_attempts: Optional[int] = None,
):
    """
    Assemble the production agent: any tool list + a model pool that rotates on
    every call and self-heals around rate limits.

    `checkpointer` defaults to `InMemorySaver()` (fine for dev/single-process).
    For real production deployments, swap in a durable one, e.g.:
        from langgraph.checkpoint.postgres import PostgresSaver
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            agent = build_agent(checkpointer=checkpointer)
    """
    pool = pool or build_default_pool()
    tools = DEFAULT_TOOLS if tools is None else tools

    return create_agent(
        model=pool.primary,  # static default; rotation middleware overrides every real call
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            make_rotating_model_middleware(pool, max_attempts=max_attempts),
            handle_tool_errors,
        ],
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )


# =============================================================================
# 6. Demo REPL
# =============================================================================

def _demo() -> None:
    import dotenv
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    agent = build_agent()
    config = {"configurable": {"thread_id": "demo-thread"}}

    print("Rotating multi-model agent ready. Type 'exit' to quit.\n")
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