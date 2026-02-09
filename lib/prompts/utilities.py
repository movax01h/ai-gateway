"""Shared prompt utilities for ai_gateway and duo_workflow_service.

This module contains utility functions for working with prompts that are used by both services and have minimal
dependencies.
"""

from typing import Sequence

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts.chat import MessageLikeRepresentation

__all__ = ["prompt_template_to_messages"]


def prompt_template_to_messages(
    tpl: dict[str, str],
) -> Sequence[MessageLikeRepresentation]:
    """Convert a prompt template dictionary to a sequence of message representations.

    Args:
        tpl: A dictionary mapping role names to content strings. If the role is
            "placeholder", it creates a MessagesPlaceholder instead of a tuple.

    Returns:
        A sequence of message-like representations suitable for ChatPromptTemplate.

    Example:
        >>> tpl = {"system": "You are helpful", "placeholder": "history", "user": "Hello"}
        >>> messages = prompt_template_to_messages(tpl)
        >>> # Returns: [("system", "You are helpful"), MessagesPlaceholder("history"), ("user", "Hello")]
    """
    return [
        MessagesPlaceholder(content) if role == "placeholder" else (role, content)
        for role, content in tpl.items()
    ]
