"""Audit capture for web searches, shared by the tool and the agent that run them."""

from typing import Any, Optional

from duo_workflow_service.audit_events.context import get_audit_collector
from duo_workflow_service.audit_events.event_types import WebSearchInvokedEvent
from duo_workflow_service.entities.server_tool_blocks import is_web_search_call
from lib.context import current_model_metadata_context


def current_model() -> tuple[str, str]:
    """Name and hosting provider (e.g. `Anthropic`, `Bedrock`) of the selected model."""
    definition = getattr(current_model_metadata_context.get(), "llm_definition", None)
    params = getattr(definition, "params", None)
    name = getattr(params, "model", None) or getattr(definition, "name", None)
    return name or "unknown", getattr(definition, "provider", None) or "unknown"


def capture_web_search_invoked(
    search_source: str, model_name: Optional[str] = None
) -> None:
    """Record one web search, when this session is audited.

    `model_name` overrides the selected model for callers holding the response's own name.
    """
    collector = get_audit_collector()
    if collector is None:
        return

    selected_name, provider = current_model()
    collector.capture(
        WebSearchInvokedEvent(
            workflow_id=collector.workflow_id,
            model_name=model_name or selected_name,
            provider=provider,
            search_source=search_source,
        )
    )


def capture_web_searches(content: Any, model_name: Optional[str] = None) -> None:
    """Record every search the model ran itself in one message's content.

    A provider-side search fires no tool callback, so nothing else observes it.
    """
    if not isinstance(content, list):
        return

    for block in content:
        if is_web_search_call(block):
            capture_web_search_invoked("native", model_name=model_name)
