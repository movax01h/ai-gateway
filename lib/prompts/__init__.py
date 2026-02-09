"""Shared prompt infrastructure for ai_gateway and duo_workflow_service.

This module exports utilities and types that are used by both services.
"""

from lib.prompts.caching import (
    X_GITLAB_MODEL_PROMPT_CACHE_ENABLED,
    current_prompt_cache_context,
    prompt_caching_enabled_in_current_request,
    set_prompt_caching_enabled_to_current_request,
)
from lib.prompts.utilities import prompt_template_to_messages

__all__ = [
    # Caching
    "X_GITLAB_MODEL_PROMPT_CACHE_ENABLED",
    "current_prompt_cache_context",
    "prompt_caching_enabled_in_current_request",
    "set_prompt_caching_enabled_to_current_request",
    # Utilities
    "prompt_template_to_messages",
]
