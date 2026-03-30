import pytest

from ai_gateway.models.base import log_request
from ai_gateway.models.v2.container import (
    ContainerModels,
    _mock_selector,
    litellm,
)


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


@pytest.mark.parametrize(
    ("duo_workflow_dict", "expected_url"),
    [
        (
            {
                "use_caching_proxy": False,
                "caching_proxy": {"url": "http://localhost:8888"},
            },
            None,
        ),
        (
            {
                "use_caching_proxy": True,
                "caching_proxy": {"url": "http://proxy.test:8888"},
            },
            "http://proxy.test:8888",
        ),
    ],
)
def test_duo_workflow_caching_proxy_url_via_container(duo_workflow_dict, expected_url):
    container = ContainerModels()
    container.config.from_dict(
        {
            "duo_workflow": duo_workflow_dict,
            "mock_model_responses": False,
            "use_agentic_mock": False,
        }
    )

    assert container._duo_workflow().caching_proxy_url() == expected_url
