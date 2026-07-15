from unittest.mock import MagicMock

import pytest
import requests

from ai_gateway.scripts import troubleshoot_selfhosted_installation as ts

MANTLE_ENDPOINT = "https://bedrock-mantle.us-east-1.api.aws/v1"


class TestCheckProviderSpecificEnvVariables:
    def test_bedrock_mantle_with_api_key_does_not_raise(self, monkeypatch, capsys):
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "a-key")

        ts.check_provider_specific_env_variables("bedrock_mantle")

        assert "BEDROCK_MANTLE_API_KEY is set" in capsys.readouterr().out

    def test_bedrock_mantle_without_api_key_does_not_raise(self, monkeypatch, capsys):
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        ts.check_provider_specific_env_variables("bedrock_mantle")

        assert "--api-key" in capsys.readouterr().out

    def test_bedrock_still_requires_aws_variables(self, monkeypatch):
        for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"):
            monkeypatch.delenv(var, raising=False)

        with pytest.raises(ValueError, match="Missing environment variables"):
            ts.check_provider_specific_env_variables("bedrock")


class TestCheckBedrockMantleAccessible:
    def test_probes_endpoint_with_bearer_token(self, monkeypatch, capsys):
        calls = {}

        def fake_get(url, headers=None, timeout=None):
            calls["url"] = url
            calls["headers"] = headers
            return object()

        monkeypatch.setattr(ts.requests, "get", fake_get)

        ts.check_bedrock_mantle_accessible(MANTLE_ENDPOINT, "a-key")

        assert calls["url"] == MANTLE_ENDPOINT
        assert calls["headers"]["Authorization"] == "Bearer a-key"
        assert "Bedrock Mantle is accessible" in capsys.readouterr().out

    def test_no_authorization_header_without_api_key(self, monkeypatch):
        calls = {}

        def fake_get(url, headers=None, timeout=None):
            calls["headers"] = headers
            return object()

        monkeypatch.setattr(ts.requests, "get", fake_get)

        ts.check_bedrock_mantle_accessible(MANTLE_ENDPOINT, None)

        assert "Authorization" not in calls["headers"]

    def test_connection_error_raises_runtime_error(self, monkeypatch):
        def fake_get(*_args, **_kwargs):
            raise requests.ConnectionError("boom")

        monkeypatch.setattr(ts.requests, "get", fake_get)

        with pytest.raises(RuntimeError, match="Bedrock Mantle endpoint"):
            ts.check_bedrock_mantle_accessible(MANTLE_ENDPOINT, "a-key")

    def test_no_endpoint_skips_probe(self, monkeypatch):
        def fail_get(*_args, **_kwargs):
            raise AssertionError("requests.get should not be called")

        monkeypatch.setattr(ts.requests, "get", fail_get)

        ts.check_bedrock_mantle_accessible(None, "a-key")


class TestCheckProviderAccessibleRouting:
    def test_routes_bedrock_mantle_to_http_probe(self, monkeypatch):
        seen = {}

        monkeypatch.setattr(
            ts,
            "check_bedrock_mantle_accessible",
            lambda endpoint, api_key: seen.update(endpoint=endpoint, api_key=api_key),
        )

        ts.check_provider_accessible("bedrock_mantle", MANTLE_ENDPOINT, "a-key")

        assert seen == {"endpoint": MANTLE_ENDPOINT, "api_key": "a-key"}

    def test_other_provider_is_a_noop(self, monkeypatch):
        def fail_get(*_args, **_kwargs):
            raise AssertionError("requests.get should not be called")

        monkeypatch.setattr(ts.requests, "get", fail_get)

        ts.check_provider_accessible("custom_openai", MANTLE_ENDPOINT, "a-key")


class TestTroubleshootWiring:
    def test_bedrock_mantle_identifier_routes_to_provider_checks(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "troubleshoot",
                "--model-family",
                "gpt",
                "--model-identifier",
                "bedrock_mantle/openai.gpt-oss-120b",
                "--api-key",
                "a-key",
            ],
        )
        for name in (
            "check_general_env_variables",
            "check_aigw_endpoint",
            "check_gitlab_connectivity",
            "check_customer_portal_reachable",
            "check_dws_health",
            "check_provider_specific_env_variables",
            "check_provider_accessible",
            "check_suggestions_model_access",
        ):
            monkeypatch.setattr(ts, name, MagicMock())

        ts.troubleshoot()

        ts.check_provider_specific_env_variables.assert_called_once_with(
            "bedrock_mantle"
        )
        ts.check_provider_accessible.assert_called_once_with(
            "bedrock_mantle", "http://localhost:4000", "a-key"
        )
