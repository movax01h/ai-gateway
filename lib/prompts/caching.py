from contextvars import ContextVar

X_GITLAB_MODEL_PROMPT_CACHE_ENABLED = "X-Gitlab-Model-Prompt-Cache-Enabled"

current_prompt_cache_context: ContextVar[str | None] = ContextVar(
    "current_prompt_cache_context", default=None
)


def set_prompt_caching_enabled_to_current_request(value: str | None):
    current_prompt_cache_context.set(value)


def prompt_caching_enabled_in_current_request() -> str:
    return current_prompt_cache_context.get() or ""
