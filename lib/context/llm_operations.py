"""LLM operations and token usage tracking context variables.

These context variables track token usage and LLM operations across request lifecycles for billing, metrics, and
debugging.
"""

from contextvars import ContextVar

type TokenUsage = dict[str, dict[str, int]]
type LlmOperations = list[dict[str, str | int | None]]

# Read token_usage non-destructively: streamed responses read it on every chunk, but
# usage is only registered once the LLM emits its usage metadata at the end of the
# stream, so a read must never reset the accumulator. Cross-request isolation comes from
# init_token_usage(), which resets it at the start of each request that tracks usage.
# See https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/2534.
token_usage: ContextVar[TokenUsage | None] = ContextVar("token_usage", default=None)
llm_operations: ContextVar[LlmOperations | None] = ContextVar(
    "llm_operations", default=None
)


def init_token_usage() -> None:
    """Initialize token usage tracking for the current context."""
    token_usage.set({})


def init_llm_operations() -> None:
    """Initialize LLM operations tracking for the current context."""
    llm_operations.set([])


def get_llm_operations() -> LlmOperations | None:
    """Get and reset LLM operations for the current context.

    Returns the current operations list and resets it to None to prevent duplicate reporting across multiple requests.
    """
    current_operations = llm_operations.get()
    # Reset the operations so multiple requests don't return the same values
    llm_operations.set(None)
    return current_operations
