from contextvars import ContextVar

__all__ = [
    "enabled_instance_verbose_ai_logs",
    "current_verbose_ai_logs_context",
    "VERBOSE_AI_LOGS_HEADER",
]

# Header key used for verbose AI logs in both HTTP and gRPC contexts
VERBOSE_AI_LOGS_HEADER = "x-gitlab-enabled-instance-verbose-ai-logs"


def enabled_instance_verbose_ai_logs() -> bool:
    """Check if instance verbose AI logs are enabled.

    This function works in both AI Gateway (HTTP/Starlette) and DWS (gRPC) contexts
    by using a shared context variable that both services can set.

    Returns:
        bool: True if instance verbose AI logs are enabled, False otherwise.
    """
    return current_verbose_ai_logs_context.get(False)


current_verbose_ai_logs_context: ContextVar[bool] = ContextVar(
    "current_verbose_ai_logs_context", default=False
)
