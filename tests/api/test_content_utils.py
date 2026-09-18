import pytest

from ai_gateway.api.content_utils import content_to_text


def test_plain_string_passes_through():
    assert content_to_text("Hi John!") == "Hi John!"


def test_joins_text_blocks():
    content = [{"type": "text", "text": "Hi "}, {"type": "text", "text": "John!"}]

    assert content_to_text(content) == "Hi John!"


def test_skips_non_text_blocks():
    content = [
        {"type": "thinking", "thinking": "Let me think..."},
        {"type": "text", "text": "Hi "},
        {"type": "tool_use", "name": "get_name", "input": {}},
        {"type": "text", "text": "John!"},
    ]

    assert content_to_text(content) == "Hi John!"


@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "thinking", "thinking": "Let me think..."}],
        # A text block without a `text` key contributes nothing
        [{"type": "text"}],
        # Bare strings inside the list are skipped: only dict blocks with
        # `type == "text"` match today. Whether they should pass through is an
        # open question, so this pins current behaviour rather than endorsing it.
        ["Hi John!"],
    ],
)
def test_flattens_to_empty_string(content):
    assert content_to_text(content) == ""


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (123, "123"),
        (None, "None"),
        ({"type": "text", "text": "Hi John!"}, "{'type': 'text', 'text': 'Hi John!'}"),
    ],
)
def test_falls_back_to_str(content, expected):
    """Content that is neither a string nor a list is stringified.

    Not reachable through the endpoints (`BaseMessage.content` only allows `str` or `list`), but it preserves the
    pre-existing `str(chunk.content)` behaviour.
    """
    assert content_to_text(content) == expected
