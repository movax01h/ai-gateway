"""Shared prompt infrastructure for ai_gateway and duo_workflow_service."""

from lib.prompts.caching import (
    X_GITLAB_MODEL_PROMPT_CACHE_ENABLED,
    current_prompt_cache_context,
    prompt_caching_enabled_in_current_request,
    set_prompt_caching_enabled_to_current_request,
)

__all__ = [
    "X_GITLAB_MODEL_PROMPT_CACHE_ENABLED",
    "current_prompt_cache_context",
    "prompt_caching_enabled_in_current_request",
    "set_prompt_caching_enabled_to_current_request",
]
