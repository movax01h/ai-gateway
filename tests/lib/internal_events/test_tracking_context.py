import pytest

from lib.internal_events.tracking_context import (
    parse_tracking_context,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        pytest.param(None, None, id="none"),
        pytest.param("", None, id="empty_string"),
        pytest.param(
            '{"distribution": "npm", "execution_environment": "CI"}',
            {"distribution": "npm", "execution_environment": "CI"},
            id="valid_payload",
        ),
        pytest.param(
            '{"distribution": "glab", "future_key": "value"}',
            {"distribution": "glab", "future_key": "value"},
            id="open_ended_keys_forwarded",
        ),
        pytest.param('{"empty": {}}', {"empty": {}}, id="nested_value_kept"),
        pytest.param("not-json", None, id="malformed_json"),
        pytest.param('{"unclosed": ', None, id="truncated_json"),
        pytest.param('["a", "b"]', None, id="json_array"),
        pytest.param('"a string"', None, id="json_string"),
        pytest.param("42", None, id="json_number"),
        pytest.param("null", None, id="json_null"),
    ],
)
def test_parse_tracking_context(raw, expected):
    assert parse_tracking_context(raw) == expected


def test_parse_tracking_context_coerces_keys_to_str():
    # JSON object keys are always strings, but assert the contract explicitly.
    assert parse_tracking_context('{"1": "one"}') == {"1": "one"}
