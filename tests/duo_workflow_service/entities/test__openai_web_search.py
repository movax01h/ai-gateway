import pytest

from duo_workflow_service.entities._openai_web_search import (
    card_fields,
    results_by_call_id,
)
from duo_workflow_service.entities.state import ToolStatus


def _url_citation(url, title):
    return {
        "type": "url_citation",
        "url": url,
        "title": title,
        "start_index": 0,
        "end_index": 1,
    }


def _answer(*citations):
    """The assistant text OpenAI hangs its ``url_citation`` annotations off."""
    return {"type": "text", "text": "Answer.", "annotations": list(citations)}


def _call(call_id="ws_1", *, status="completed", sources=None, results=None, **action):
    """An OpenAI ``web_search_call``; ``action`` is omitted entirely when not described."""
    block = {"type": "web_search_call", "id": call_id, "status": status}
    if action or sources is not None:
        block["action"] = {**action}
        if sources is not None:
            block["action"]["sources"] = sources
    if results is not None:
        block["results"] = results
    return block


def _source(url, title=None):
    source = {"type": "url", "url": url}
    if title is not None:
        source["title"] = title
    return source


def _result(url, title=None):
    result = {"type": "web_search_result", "url": url}
    if title is not None:
        result["title"] = title
    return result


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(
            [
                _call(type="search", query="q"),
                _answer(
                    _url_citation("https://a", "A"), _url_citation("https://a", "A")
                ),
            ],
            {"ws_1": [_result("https://a", "A")]},
            id="citations-become-results-deduped",
        ),
        pytest.param(
            [
                _call(
                    "ws_1", type="search", query="q1", sources=[_source("https://a")]
                ),
                _call(
                    "ws_2", type="search", query="q2", sources=[_source("https://b")]
                ),
                _answer(
                    _url_citation("https://a", "A"), _url_citation("https://b", "B")
                ),
            ],
            {"ws_1": [_result("https://a", "A")], "ws_2": [_result("https://b", "B")]},
            id="action-sources-attribute-results-per-call",
        ),
        pytest.param(
            # Without `sources`, back-to-back searches cannot be told apart.
            [_call("ws_1"), _call("ws_2"), _answer(_url_citation("https://a", "A"))],
            {"ws_1": [], "ws_2": [_result("https://a", "A")]},
            id="citations-fall-to-the-most-recent-call",
        ),
        pytest.param(
            # OpenAI cites the page with a `utm_source` its `action.sources` omits.
            [
                _call(
                    type="search", query="q", sources=[_source("https://a/page?id=1")]
                ),
                _answer(_url_citation("https://a/page?id=1&utm_source=openai", "A")),
            ],
            {"ws_1": [_result("https://a/page?id=1", "A")]},
            id="tracked-citation-titles-untracked-source",
        ),
        pytest.param(
            # `action.sources` names every match but titles none, and the answer cites only
            # one. `results` is what titles the rest; the remainder stays untitled.
            [
                _call(
                    type="search",
                    query="q",
                    sources=[
                        _source("https://cited"),
                        _source("https://read"),
                        _source("https://matched-only"),
                    ],
                    results=[
                        {
                            "url": "https://read?utm_source=openai",
                            "title": "Read",
                            "snippet": "...",
                        }
                    ],
                ),
                _answer(_url_citation("https://cited", "Cited")),
            ],
            {
                "ws_1": [
                    _result("https://cited", "Cited"),
                    _result("https://read", "Read"),
                    _result("https://matched-only"),
                ]
            },
            id="results-title-sources-the-answer-never-cited",
        ),
        pytest.param(
            [
                _call(type="search", sources=[_source("https://a", "From source")]),
                _answer(_url_citation("https://a", "From citation")),
            ],
            {"ws_1": [_result("https://a", "From source")]},
            id="source-title-wins-over-citation-title",
        ),
        pytest.param(
            [
                _call(),
                {"type": "reasoning", "annotations": [_url_citation("https://a", "A")]},
            ],
            {"ws_1": []},
            id="annotations-outside-a-text-block-are-ignored",
        ),
        pytest.param(
            [_call(), _answer("not-a-dict", {"type": "url_citation"})],
            {"ws_1": []},
            id="citations-naming-no-url-are-skipped",
        ),
        pytest.param(
            [_call(sources=[_source("http://[")])],
            {"ws_1": [_result("http://[")]},
            id="unparseable-url-is-kept-verbatim",
        ),
        pytest.param(
            [{"type": "web_search_call", "id": "ws_1", "action": "not-a-dict"}],
            {"ws_1": []},
            id="action-of-another-shape",
        ),
        pytest.param(
            # Incomplete `sources`: the answer cites a page the search never listed.
            [
                _call(type="search", sources=[_source("https://listed")]),
                _answer(
                    _url_citation("https://listed", "Listed"),
                    _url_citation("https://unlisted", "Unlisted"),
                ),
            ],
            {
                "ws_1": [
                    _result("https://listed", "Listed"),
                    _result("https://unlisted", "Unlisted"),
                ]
            },
            id="citations-no-call-claims-are-not-lost",
        ),
        pytest.param(
            # The second answer repeats a citation the first call already took.
            [
                _call("ws_1"),
                _answer(_url_citation("https://a", "A")),
                _call("ws_2", type="search", sources=[_source("https://b")]),
                _answer(_url_citation("https://a", "A")),
            ],
            {"ws_1": [_result("https://a", "A")], "ws_2": [_result("https://b")]},
            id="a-citation-stays-with-the-call-that-claimed-it",
        ),
        pytest.param(
            # A page read carries no `sources`, so its citations reach it by the fallback.
            [
                _call(type="open_page", url="https://a"),
                _answer(_url_citation("https://a", "A")),
            ],
            {"ws_1": [_result("https://a", "A")]},
            id="page-reads-are-attributed-through-citations",
        ),
        pytest.param(
            [
                {"type": "web_search_call", "action": {"type": "search"}},
                _call("ws_2"),
                _answer(_url_citation("https://a", "A")),
            ],
            {"ws_2": [_result("https://a", "A")]},
            id="calls-without-an-id-are-skipped",
        ),
    ],
)
def test_results_by_call_id(content, expected):
    assert results_by_call_id(content) == expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("completed", ToolStatus.SUCCESS),
        ("failed", ToolStatus.FAILURE),
        ("in_progress", ToolStatus.PENDING),
        (None, ToolStatus.PENDING),
    ],
)
def test_card_fields_status_mapping(status, expected):
    _, _, status_out = card_fields(_call(status=status))

    assert status_out == expected


@pytest.mark.parametrize(
    ("action", "expected_name", "expected_args"),
    [
        ({"type": "search", "query": "q"}, "web_search", {"query": "q"}),
        ({"type": "open_page", "url": "https://x"}, "web_fetch", {"url": "https://x"}),
        (
            {"type": "find_in_page", "url": "https://x", "pattern": "p"},
            "web_fetch",
            {"url": "https://x", "pattern": "p"},
        ),
        ({}, "web_search", {}),
        pytest.param(
            # Observed from gpt-5.6-sol: one call, two searches, `query` holding only the
            # first of them.
            {
                "type": "search",
                "query": "latest news",
                "queries": ["latest news", "breaking news today"],
            },
            "web_search",
            {"query": "latest news, breaking news today"},
            id="several-queries-in-one-call-are-all-kept",
        ),
        pytest.param(
            {"type": "search", "queries": ["only plural"]},
            "web_search",
            {"query": "only plural"},
            id="queries-without-query",
        ),
        pytest.param(
            {"type": "search", "query": "q", "queries": [None, 7]},
            "web_search",
            {"query": "q"},
            id="malformed-queries-fall-back-to-query",
        ),
        pytest.param("not-a-dict", "web_search", {}, id="action-of-another-shape"),
        pytest.param(None, "web_search", {}, id="no-action"),
    ],
)
def test_card_fields_from_action(action, expected_name, expected_args):
    name, args, _ = card_fields(
        {"type": "web_search_call", "id": "ws_1", "action": action}
    )

    assert (name, args) == (expected_name, expected_args)
