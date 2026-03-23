from contextvars import ContextVar

X_GITLAB_LANGSMITH_TRACE_HEADER = "langsmith-trace"

langsmith_trace_headers: ContextVar[dict[str, str] | None] = ContextVar(
    "langsmith_trace_headers", default=None
)


def set_langsmith_trace_headers(headers: dict[str, str] | None):
    """Set the LangSmith trace headers for the current request."""
    langsmith_trace_headers.set(headers)


def get_langsmith_trace_headers() -> dict[str, str] | None:
    """Get the LangSmith trace headers for the current request."""
    return langsmith_trace_headers.get()
