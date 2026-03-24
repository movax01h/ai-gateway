"""Shared prompt utilities for ai_gateway and duo_workflow_service.

This module contains utility functions for working with prompts that are used by both services and have minimal
dependencies.
"""

from importlib import resources
from typing import Sequence

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts.chat import MessageLikeRepresentation

__all__ = ["TOOL_OUTPUT_SECURITY_INCLUDE", "prompt_template_to_messages"]

TOOL_OUTPUT_SECURITY_INCLUDE = (
    resources.files("ai_gateway.prompts.definitions")
    .joinpath("common", "tool_output_security", "1.0.0.jinja")
    .read_text()
    + "\n\n"
)


def prompt_template_to_messages(
    tpl: dict[str, str],
) -> Sequence[MessageLikeRepresentation]:
    """Convert a prompt template dictionary to a sequence of message representations.

    Automatically prepends the tool output security instruction to the first system message.

    Args:
        tpl: A dictionary mapping role names to content strings. If the role is
            "placeholder", it creates a MessagesPlaceholder instead of a tuple.

    Returns:
        A sequence of message-like representations suitable for ChatPromptTemplate.

    Example:
        >>> tpl = {"system": "You are helpful", "placeholder": "history", "user": "Hello"}
        >>> messages = prompt_template_to_messages(tpl)
        >>> # Returns: [("system", "<security>You are helpful"), MessagesPlaceholder("history"), ("user", "Hello")]
    """
    messages: list[MessageLikeRepresentation] = []
    security_injected = False
    for role, content in tpl.items():
        if role == "placeholder":
            messages.append(MessagesPlaceholder(content))
        else:
            if not security_injected and role.startswith("system"):
                content = TOOL_OUTPUT_SECURITY_INCLUDE + content
                security_injected = True
            messages.append((role, content))
    return messages
