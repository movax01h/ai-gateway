import httpx
import pytest
from anthropic import Anthropic, AsyncAnthropic

from ai_gateway.models.v2 import ChatAnthropic


class TestChatAnthropic:
    @pytest.mark.parametrize(
        ("model_options", "expected_options"),
        [
            (
                {},
                {
                    "default_request_timeout": 60.0,
                    "max_retries": 1,
                    "default_headers": {"anthropic-version": "2023-06-01"},
                },
            ),
            (
                {
                    "default_request_timeout": 10,
                    "max_retries": 2,
                    "default_headers": {"anthropic-version": "2021-06-01"},
                },
                {
                    "default_request_timeout": 10,
                    "max_retries": 2,
                    "default_headers": {"anthropic-version": "2021-06-01"},
                },
            ),
        ],
    )
    def test_async_model_options(self, model_options: dict, expected_options: dict):
        model = ChatAnthropic(
            async_client=AsyncAnthropic(),
            model="claude-3-5-sonnet-20241022",
            **model_options,
        )  # type: ignore[call-arg]

        assert isinstance(model._async_client, AsyncAnthropic)
        assert (
            model._async_client.timeout == expected_options["default_request_timeout"]
        )
        assert model._async_client.max_retries == expected_options["max_retries"]

        all_headers = [
            model._async_client.default_headers[h_key] == h_value
            for h_key, h_value in expected_options["default_headers"].items()
        ]
        assert all(all_headers)

    def test_unsupported_sync_methods(self):
        model = ChatAnthropic(
            async_client=AsyncAnthropic(), model="claude-3-5-sonnet-20241022"
        )  # type: ignore[call-arg]

        with pytest.raises(NotImplementedError):
            model.invoke("What's your name?")

    def test_overwrite_anthropic_credentials(self):
        model = ChatAnthropic(
            async_client=AsyncAnthropic(),
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key="test_api_key",
            anthropic_api_url="http://anthropic.test",
        )  # type: ignore[call-arg]

        assert model.anthropic_api_key.get_secret_value() == "test_api_key"
        assert model._async_client.api_key == "test_api_key"

        assert model.anthropic_api_url == "http://anthropic.test"
        assert model._async_client.base_url == "http://anthropic.test"

    @pytest.mark.parametrize(
        ("betas", "expected_header"),
        [
            (["beta1"], "beta1"),
            (["beta1", "beta2"], "beta1,beta2"),
            (["extended-cache-ttl-2025-04-11"], "extended-cache-ttl-2025-04-11"),
            ([], None),
            (None, None),
        ],
    )
    def test_betas_configuration(self, betas, expected_header):
        model = ChatAnthropic(
            async_client=AsyncAnthropic(),
            model="claude-3-5-sonnet-20241022",
            betas=betas,
        )

        if expected_header:
            assert (
                model._async_client.default_headers["anthropic-beta"] == expected_header
            )
        else:
            assert model._async_client.default_headers.get("anthropic-beta") is None

    def test_get_combined_headers_method(self):
        model = ChatAnthropic(
            async_client=AsyncAnthropic(),
            model="claude-3-5-sonnet-20241022",
        )

        headers = model._get_combined_headers()
        assert headers == {"anthropic-version": "2023-06-01"}

        model_with_betas = ChatAnthropic(
            async_client=AsyncAnthropic(),
            model="claude-3-5-sonnet-20241022",
            betas=["beta1", "beta2"],
        )

        headers_with_betas = model_with_betas._get_combined_headers()
        expected_headers = {
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "beta1,beta2",
        }
        assert headers_with_betas == expected_headers
