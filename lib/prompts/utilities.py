"""Shared prompt utilities for ai_gateway and duo_workflow_service.

This module contains utility functions for working with prompts that are used by both services and have minimal
dependencies.
"""

import hashlib
import os
from contextvars import ContextVar
from importlib import resources
from typing import Sequence, cast

from jinja2 import BaseLoader, Environment
from langchain_core.messages import SystemMessage
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts.chat import MessageLikeRepresentation

from lib.context.workflow import get_workflow_id

__all__ = [
    "merge_leading_system_messages",
    "prompt_template_to_messages",
    "render_security_block",
]

_SECURITY_TEMPLATE_SOURCE = (
    resources.files("ai_gateway.prompts.definitions")
    .joinpath("common", "tool_output_security", "1.0.0.jinja")
    .read_text()
)

# Template source is package-internal and not user-controlled.
_jinja_env = Environment(loader=BaseLoader())
_security_template = _jinja_env.from_string(_SECURITY_TEMPLATE_SOURCE)

_security_suffix_var: ContextVar[str | None] = ContextVar(
    "security_suffix", default=None
)


def _security_suffix() -> str:
    """Return a per-session suffix for security block delimiters.

    For workflows with workflow_id (chat sessions), derives suffix from workflow_id so it remains stable across turns in
    same chat (enables prompt caching).

    Otherwise generates random suffix on first call within context and caches it.
    """
    suffix = _security_suffix_var.get()
    if suffix is None:
        # Try derive from workflow_id for stable suffix across chat turns
        workflow_id = get_workflow_id()
        if workflow_id:
            # Deterministic suffix from workflow_id
            suffix = hashlib.sha256(workflow_id.encode()).hexdigest()[:16]
        else:
            # Fallback to random for non-workflow contexts
            suffix = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
        # Safe to cache without explicit cleanup: each ASGI request and each
        # gRPC handler runs in its own contextvars.Context copy, so this value
        # is naturally isolated per-request/per-session without manual clearing.
        _security_suffix_var.set(suffix)
    return suffix


def render_security_block() -> str:
    """Render the tool output security block with a randomized per-session suffix.

    Returns:
        The rendered security block string.
    """
    suffix = _security_suffix()
    return _security_template.render(suffix=suffix) + "\n\n"


def _is_system_message(message: MessageLikeRepresentation) -> bool:
    return isinstance(message, tuple) and str(message[0]).startswith("system")


def _insert_security_block(messages: list[MessageLikeRepresentation]) -> None:
    """Insert the security block after the run of system messages that starts with the first one.

    Prompts without a system message get no block, as before.
    """
    first = next((i for i, m in enumerate(messages) if _is_system_message(m)), None)
    if first is None:
        return

    end = first
    while end < len(messages) and _is_system_message(messages[end]):
        end += 1

    # As its own message the block needs no separator after it; Jinja would drop one trailing newline anyway.
    messages.insert(end, ("system", render_security_block().rstrip("\n")))


def merge_leading_system_messages(prompt: PromptValue) -> PromptValue:
    """Merge the leading run of system messages into one, for models that accept only one system message.

    Some chat templates (for example Qwen 3.x) reject any system message that is not the first message. Merging after
    rendering, instead of joining template strings, keeps text that Jinja would drop after an ``{% extends %}`` block.
    """
    messages = prompt.to_messages()
    end = 0
    while end < len(messages) and isinstance(messages[end], SystemMessage):
        end += 1

    if end < 2:
        return prompt

    merged = SystemMessage(content="\n\n".join(m.text for m in messages[:end]))
    return ChatPromptValue(messages=[merged, *messages[end:]])


def prompt_template_to_messages(
    tpl: dict[str, str | list[str]],
    tool_output_security: bool = True,
) -> Sequence[MessageLikeRepresentation]:
    """Convert a prompt template dictionary to a sequence of message representations.

    Automatically inserts the tool output security instruction as its own system message, directly after the leading
    run of system messages. The block carries a per-session suffix, so keeping it out of the author's system messages
    lets the system prefix stay byte-identical across sessions and be read from the prompt cache.
    Automatically appends an optional ``MessagesPlaceholder("history")`` when not already
    present, to prevent silent infinite loops when AgentNode passes history but the template
    does not declare it.

    Args:
        tpl: A dictionary mapping role names to content strings or lists of strings. If the
            role is "placeholder", it creates a MessagesPlaceholder instead of a tuple. When
            a role maps to a list of strings, each item becomes a separate message with that
            role, enabling multiple system messages for prompt caching purposes.
        tool_output_security: When True, insert the tool-output security block as a system
            message after the leading system messages. Set False for prompts without tool
            output, such as code completion.

    Returns:
        A sequence of message-like representations suitable for ChatPromptTemplate.

    Example:
        >>> tpl = {"system": "You are helpful", "placeholder": "history", "user": "Hello"}
        >>> messages = prompt_template_to_messages(tpl)
        >>> Returns: [("system", "You are helpful"), ("system", "<security>"), MessagesPlaceholder("history"),
        ...           ("user", "Hello")]

        >>> tpl = {"system": ["Static part", "Dynamic part"], "user": "Hello"}
        >>> messages = prompt_template_to_messages(tpl)
        >>> Returns: [("system", "Static part"), ("system", "Dynamic part"), ("system", "<security>"),
        ...           ("user", "Hello"), ...]
    """
    messages: list[MessageLikeRepresentation] = []
    for role, content in tpl.items():
        if role == "placeholder":
            messages.append(MessagesPlaceholder(cast(str, content)))
        else:
            msgs = [content] if isinstance(content, str) else content
            messages.extend((role, msg) for msg in msgs)

    if tool_output_security:
        _insert_security_block(messages)

    # Automatically add optional history placeholder if not present to prevent
    # silent infinite loops when AgentNode passes history but template doesn't declare it
    if not any(
        isinstance(m, MessagesPlaceholder) and m.variable_name == "history"
        for m in messages
    ):
        messages.append(MessagesPlaceholder("history", optional=True))

    return messages
