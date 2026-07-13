from contextvars import ContextVar

__all__ = [
    "VERBOSE_AI_LOGS_HEADER",
    "X_GITLAB_EXTENDED_LOGGING_HEADER",
    "current_verbose_ai_logs_context",
    "enabled_instance_verbose_ai_logs",
    "extended_logging_context",
    "is_extended_logging_enabled",
]

# Header key used for verbose AI logs in both HTTP and gRPC contexts
VERBOSE_AI_LOGS_HEADER = "x-gitlab-enabled-instance-verbose-ai-logs"

# Per-user/namespace extended logging header injected by Rails at /ws pre-auth time
X_GITLAB_EXTENDED_LOGGING_HEADER = "x-gitlab-extended-logging"


def enabled_instance_verbose_ai_logs() -> bool:
    """Check if instance verbose AI logs are enabled.

    This function works in both AI Gateway (HTTP/Starlette) and DWS (gRPC) contexts
    by using a shared context variable that both services can set.

    Returns:
        bool: True if instance verbose AI logs are enabled, False otherwise.
    """
    return current_verbose_ai_logs_context.get(False)


def is_extended_logging_enabled() -> bool:
    """Check if per-user/namespace extended logging is enabled.

    Set by MetadataContextInterceptor from the x-gitlab-extended-logging gRPC
    metadata header, which Rails computes and injects at /ws pre-auth time.

    Returns:
        bool: True if extended logging is enabled for this request, False otherwise.
    """
    return extended_logging_context.get(False)


current_verbose_ai_logs_context: ContextVar[bool] = ContextVar(
    "current_verbose_ai_logs_context", default=False
)

# Distinct from current_verbose_ai_logs_context, which reflects the instance-level
# x-gitlab-enabled-instance-verbose-ai-logs flag.
extended_logging_context: ContextVar[bool] = ContextVar(
    "extended_logging_context", default=False
)
