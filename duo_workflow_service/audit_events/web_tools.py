"""Audit and usage capture for web searches and fetches, shared by the tool and the agent that run them."""

from typing import Any, Optional

from duo_workflow_service.audit_events.context import get_audit_collector
from duo_workflow_service.audit_events.event_types import (
    WebFetchInvokedEvent,
    WebSearchInvokedEvent,
)
from duo_workflow_service.entities.server_tool_blocks import (
    FIND_IN_PAGE_TOOL_NAME,
    WEB_FETCH_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
    fetched_url,
    native_web_call,
    native_web_call_succeeded,
)
from duo_workflow_service.internal_events.events_utils import track_web_tool_invoked
from lib.context import current_model_metadata_context

NATIVE_SEARCH_SOURCE = "native"
AGENTCORE_SEARCH_SOURCE = "agentcore"


def current_model() -> tuple[str, str]:
    """Name and hosting provider (e.g. `Anthropic`, `Bedrock`) of the selected model."""
    definition = getattr(current_model_metadata_context.get(), "llm_definition", None)
    params = getattr(definition, "params", None)
    name = getattr(params, "model", None) or getattr(definition, "name", None)
    return name or "unknown", getattr(definition, "provider", None) or "unknown"


def capture_native_web_calls(content: Any, model_name: Optional[str] = None) -> None:
    """Record every search and fetch the model ran itself in one message's content.

    A provider-side search fires no tool callback, so nothing else observes it. Each call is
    counted as a usage event and, when audit events are on, captured as an audit event too.
    `model_name` overrides the selected model for callers holding the response's own name.
    """
    if not isinstance(content, list):
        return

    selected_name, provider = current_model()
    model_name = model_name or selected_name
    collector = get_audit_collector()
    fetched: set[str] = set()

    for block in content:
        name = native_web_call(block)
        if name == WEB_SEARCH_TOOL_NAME:
            track_web_tool_invoked(
                WEB_SEARCH_TOOL_NAME,
                NATIVE_SEARCH_SOURCE,
                native_web_call_succeeded(block, content),
                model_name,
                provider,
            )
            if collector is not None:
                collector.capture(
                    WebSearchInvokedEvent(
                        workflow_id=collector.workflow_id,
                        model_name=model_name,
                        provider=provider,
                        search_source=NATIVE_SEARCH_SOURCE,
                    )
                )
        elif name in (WEB_FETCH_TOOL_NAME, FIND_IN_PAGE_TOOL_NAME):
            url = fetched_url(block)
            # OpenAI `find_in_page` searches a page; it is a fetch only on a URL not yet read.
            if name == FIND_IN_PAGE_TOOL_NAME and (not url or url in fetched):
                continue
            if url:
                fetched.add(url)
            track_web_tool_invoked(
                WEB_FETCH_TOOL_NAME,
                NATIVE_SEARCH_SOURCE,
                native_web_call_succeeded(block, content),
                model_name,
                provider,
            )
            if collector is not None:
                collector.capture(
                    WebFetchInvokedEvent(
                        workflow_id=collector.workflow_id,
                        model_name=model_name,
                        provider=provider,
                        url=url,
                    )
                )
