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

    def test_bedrock_does_not_raise(self, monkeypatch):
        for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"):
            monkeypatch.delenv(var, raising=False)

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

    def test_bedrock_uses_default_credential_chain(self, monkeypatch, capsys):
        seen = {}
        mock_client = MagicMock()

        def fake_boto3_client(*args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return mock_client

        monkeypatch.setattr(ts.boto3, "client", fake_boto3_client)

        ts.check_provider_accessible("bedrock")

        assert seen == {"args": ("bedrock",), "kwargs": {}}
        mock_client.list_foundation_models.assert_called_once()
        assert "Provider Bedrock is accessible" in capsys.readouterr().out

    def test_bedrock_raises_runtime_error_when_inaccessible(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.list_foundation_models.side_effect = Exception("boom")

        monkeypatch.setattr(ts.boto3, "client", lambda *args, **kwargs: mock_client)

        with pytest.raises(
            RuntimeError, match="An error occurred while contacting provider bedrock"
        ):
            ts.check_provider_accessible("bedrock")


class TestCheckSuggestionsModelAccess:
    ARGS = (
        "localhost:5052",
        "mistral",
        "http://localhost:4000/v1",
        "a-key",
        "custom_openai/mistral-7b",
        "custom_openai",
    )

    def test_posts_a_v4_generation_request(self, monkeypatch, capsys):
        calls = {}

        def fake_post(url, json=None, **_):
            calls["url"] = url
            calls["json"] = json
            return MagicMock(status_code=200)

        monkeypatch.setattr(ts.requests, "post", fake_post)

        ts.check_suggestions_model_access(*self.ARGS)

        assert calls["url"] == "http://localhost:5052/v4/code/suggestions"
        component = calls["json"]["prompt_components"][0]
        assert component["type"] == "code_editor_generation"
        assert component["payload"]["file_name"] == "test.py"
        assert component["payload"]["stream"] is False
        assert calls["json"]["model_metadata"] == {
            "provider": "openai",
            "name": "mistral",
            "endpoint": "http://localhost:4000/v1",
            "api_key": "a-key",
            "identifier": "custom_openai/mistral-7b",
        }
        assert "Successfully accessed the mistral model" in capsys.readouterr().out

    def test_non_200_raises_runtime_error_with_status(self, monkeypatch):
        monkeypatch.setattr(
            ts.requests,
            "post",
            lambda *a, **k: MagicMock(status_code=422, text="Validation error"),
        )

        with pytest.raises(RuntimeError, match="422"):
            ts.check_suggestions_model_access(*self.ARGS)

    def test_connection_error_raises_runtime_error(self, monkeypatch):
        def boom(*a, **k):
            raise requests.ConnectionError("no route")

        monkeypatch.setattr(ts.requests, "post", boom)

        with pytest.raises(RuntimeError, match="no route"):
            ts.check_suggestions_model_access(*self.ARGS)


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
