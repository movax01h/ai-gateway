"""Rebuild OpenAI web-search results in Anthropic's shape, so one card builder serves both.

OpenAI emits no result block: what a search found is split across ``action.sources``,
``results`` and ``url_citation`` annotations.
"""

from typing import Any, Iterator, Optional, TypeGuard
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from duo_workflow_service.entities.state import ToolStatus

__all__ = ["card_fields", "is_call_block", "is_known_action", "results_by_call_id"]

_STATUSES: dict[str, ToolStatus] = {
    "completed": ToolStatus.SUCCESS,
    "failed": ToolStatus.FAILURE,
}

# Three actions under one block type; reading a page and searching within it are one card.
_ACTION_TOOL_NAMES: dict[str, str] = {
    "search": "web_search",
    "open_page": "web_fetch",
    "find_in_page": "web_fetch",
}


def is_call_block(block: Any) -> TypeGuard[dict]:
    return isinstance(block, dict) and block.get("type") == "web_search_call"


def is_known_action(call_block: dict) -> bool:
    """Whether the action maps to a tool name, rather than falling back to ``web_search``."""
    return _action(call_block).get("type", "") in _ACTION_TOOL_NAMES


def _action(call_block: dict) -> dict:
    action = call_block.get("action")
    return action if isinstance(action, dict) else {}


def _url_citations(block: Any) -> Iterator[dict]:
    if not (isinstance(block, dict) and block.get("type") == "text"):
        return
    for annotation in block.get("annotations") or []:
        if isinstance(annotation, dict) and annotation.get("type") == "url_citation":
            yield annotation


def _url_key(url: str) -> str:
    """Comparable form of a URL, with tracking parameters stripped.

    Citations add a ``utm_source`` that ``action.sources`` omits, so raw URLs never match.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return url

    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.startswith("utm_")
        ]
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def _url_title_pairs(items: Any) -> Iterator[tuple[str, str]]:
    for item in items:
        if isinstance(item, dict) and item.get("url") and item.get("title"):
            yield item["url"], item["title"]


def _page_titles(content: list) -> dict[str, str]:
    """Map each URL to a title, taken from the pages the model read and from the citations."""
    titles: dict[str, str] = {}
    for block in content:
        items = (
            (block.get("results") or [])
            if is_call_block(block)
            else _url_citations(block)
        )
        for url, title in _url_title_pairs(items):
            titles.setdefault(_url_key(url), title)
    return titles


def _merge_items_into_results(
    results: dict[str, dict], items: Any, titles: dict[str, str]
) -> None:
    for item in items:
        if not (isinstance(item, dict) and item.get("url")):
            continue
        key = _url_key(item["url"])
        if key in results:
            continue
        result = {"type": "web_search_result", "url": item["url"]}
        title = item.get("title") or titles.get(key)
        if title:
            result["title"] = title
        results[key] = result


def results_by_call_id(content: list) -> dict[str, list[dict]]:
    """The sources each call found, keyed by call id."""
    titles = _page_titles(content)

    by_call: dict[str, dict[str, dict]] = {}
    claimed: set[str] = set()
    last: Optional[dict[str, dict]] = None

    for block in content:
        if is_call_block(block):
            if not block.get("id"):
                continue
            results = by_call.setdefault(block["id"], {})
            _merge_items_into_results(
                results, _action(block).get("sources") or [], titles
            )
            claimed.update(results)
            last = results
        elif last is not None:
            orphans = [
                citation
                for citation in _url_citations(block)
                if _url_key(citation.get("url") or "") not in claimed
            ]
            _merge_items_into_results(last, orphans, titles)
            claimed.update(last)

    return {call_id: list(results.values()) for call_id, results in by_call.items()}


def _args(action: dict) -> dict:
    """The card's arguments for one action."""
    args = {key: action[key] for key in ("query", "url", "pattern") if action.get(key)}
    queries = [query for query in action.get("queries") or [] if isinstance(query, str)]
    if queries:
        args["query"] = ", ".join(queries)
    return args


def _tool_name(action: dict) -> str:
    """The card name for an action, defaulting to ``web_search``."""
    return _ACTION_TOOL_NAMES.get(action.get("type", ""), "web_search")


def card_fields(call_block: dict) -> tuple[str, dict, ToolStatus]:
    """The ``(name, args, status)`` a TOOL card takes from one call block."""
    action = _action(call_block)
    return (
        _tool_name(action),
        _args(action),
        _STATUSES.get(call_block.get("status", ""), ToolStatus.PENDING),
    )
