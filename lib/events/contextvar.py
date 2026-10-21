from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

X_GITLAB_SELF_HOSTED_DAP_BILLING_ENABLED = "x-gitlab-self-hosted-dap-billing-enabled"

self_hosted_dap_billing_enabled: ContextVar[bool] = ContextVar(
    "current_self_hosted_dap_billing_enabled", default=False
)

# True while a caller repeats an LLM call that failed. Self-hosted billing reads it to bill the call only once.
llm_call_is_retry: ContextVar[bool] = ContextVar(
    "current_llm_call_is_retry", default=False
)


def set_self_hosted_dap_billing_enabled(value: str) -> None:
    enabled = value.lower() in ("true", "1")
    self_hosted_dap_billing_enabled.set(enabled)


@contextmanager
def mark_llm_call_retry(is_retry: bool) -> Iterator[None]:
    """Set `llm_call_is_retry` for the LLM call made inside the block, then restore it."""
    token = llm_call_is_retry.set(is_retry)
    try:
        yield
    finally:
        llm_call_is_retry.reset(token)
