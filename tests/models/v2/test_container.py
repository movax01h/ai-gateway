import pytest

from ai_gateway.models.base import log_request
from ai_gateway.models.v2.container import _mock_selector, litellm


def test_litellm_override():
    assert "request" in litellm.module_level_aclient.event_hooks
    assert litellm.module_level_aclient.event_hooks["request"] == [log_request]


@pytest.mark.parametrize(
    ("mock_model_responses", "use_agentic_mock", "expected_selector"),
    [
        (False, False, "original"),
        (True, False, "mocked"),
        (True, True, "agentic"),
        (
            False,
            True,
            "original",
        ),  # use_agentic_mock has no effect when mock_model_responses is False
    ],
)
def test_mock_selector(mock_model_responses, use_agentic_mock, expected_selector):
    result = _mock_selector(mock_model_responses, use_agentic_mock)
    assert result == expected_selector
