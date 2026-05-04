# pylint: disable=file-naming-for-tests,unused-argument
import json
from typing import Optional
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sse_starlette.sse import AppStatus

from ai_gateway.api.v4 import api_router
from ai_gateway.api.v4.code.typing import StreamEvent
from lib.feature_flags.context import current_feature_flag_context


@pytest.fixture(autouse=True)
def reset_sse_starlette_appstatus_event():
    # To avoid RuntimeError "bound to a different event loop" during SSE streaming in parameterized tests
    AppStatus.should_exit_event = None


@pytest.fixture(name="route")
def route_fixture():
    return "/code/suggestions"


@pytest.fixture(name="fast_api_router", scope="class")
def fast_api_router_fixture():
    return api_router


@pytest.fixture(name="scopes")
def scopes_fixture():
    return ["complete_code", "generate_code", "amazon_q_integration"]


class TestEditorContentCompletionStream:
    def test_successful_stream_response(
        self,
        mock_client: TestClient,
        mock_completions_stream: Mock,
        mock_suggestions_output_text: str,
        route: str,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "model_provider": "anthropic",
            "stream": True,
            "model_name": "claude-sonnet-4-5-20250929",
        }

        prompt_component = {
            "type": "code_editor_completion",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
        }

        current_feature_flag_context.set({"flag_a", "flag_b"})

        response = mock_client.post(
            route,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
                "X-Gitlab-Global-User-Id": "test-user-id",
            },
            json=data,
        )

        expected_model_metadata = {
            "engine": "agent",
            "name": "Codestral 22B Code Completions",
        }

        expected_feature_flags = {"flag_a", "flag_b"}

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert response.headers["X-Streaming-Format"] == "sse"

        _assert_stream_sse_responses(
            response.text,
            mock_suggestions_output_text,
            expected_model_metadata,
            expected_feature_flags,
        )


class TestEditorContentGenerationStream:
    @pytest.mark.parametrize(
        (
            "prompt_id",
            "model_provider",
            "prompt_version",
        ),
        [
            (
                None,
                "vertex-ai",
                None,
            ),
            (
                None,
                "anthropic",
                None,
            ),
            (
                "code_suggestions/generations",
                None,
                "1.1.0-dev",
            ),
            (
                "code_suggestions/generations",
                None,
                "1.2.0-dev",
            ),
            (
                "code_suggestions/generations",
                None,
                "1.2.0",
            ),
        ],
    )
    def test_successful_stream_response(
        self,
        mock_client: TestClient,
        mock_generations_stream: Mock,
        mock_suggestions_output_text: str,
        route: str,
        prompt_id: Optional[str],
        model_provider: Optional[str],
        prompt_version: Optional[str],
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "prompt_id": prompt_id,
            "model_provider": model_provider,
            "prompt_version": prompt_version,
            "stream": True,
        }

        prompt_component = {
            "type": "code_editor_generation",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
        }

        current_feature_flag_context.set({"flag_a", "flag_b"})

        response = mock_client.post(
            route,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
                "X-Gitlab-Global-User-Id": "test-user-id",
            },
            json=data,
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert response.headers["X-Streaming-Format"] == "sse"

        _assert_stream_sse_responses(
            response.text,
            mock_suggestions_output_text,
            {"engine": "agent", "name": "Code Generations Agent"},
            {"flag_a", "flag_b"},
        )


def _assert_stream_sse_responses(
    response_text: str,
    expected_suggestions_output_text: str,
    expected_model_metadata: dict,
    expected_feature_flags: set,
):
    def _parse_sse_messages():
        parsed_list = []
        for message in response_text.strip().split("\r\n\r\n"):
            lines = message.splitlines()
            event = lines[0].removeprefix("event: ")
            data = json.loads(lines[1].removeprefix("data: "))
            parsed_list.append({"event": event, "data": data})
        return parsed_list

    sse_messages = _parse_sse_messages()
    start_message = sse_messages.pop(0)
    end_message = sse_messages.pop()

    assert start_message["event"] == StreamEvent.START
    assert start_message["data"]["metadata"]["model"] == expected_model_metadata
    assert start_message["data"]["metadata"]["timestamp"] > 0
    assert start_message["data"]["metadata"]["region"] == "us-central1"
    assert (
        set(start_message["data"]["metadata"]["enabled_feature_flags"])
        == expected_feature_flags
    )

    assert end_message["event"] == StreamEvent.END
    assert end_message["data"] is None

    # _mock_async_execute() yields one character at a time, so we expect
    # a content chunk message for each character of the output text.
    assert len(expected_suggestions_output_text) == len(sse_messages)

    for index, content_message in enumerate(sse_messages):
        assert content_message["event"] == StreamEvent.CONTENT_CHUNK
        assert content_message["data"]["choices"][0]["index"] == 0
        assert (
            content_message["data"]["choices"][0]["delta"]["content"]
            == expected_suggestions_output_text[index]
        )


class TestCustomModelsSSRF:
    @pytest.fixture(name="config_values")
    def config_values_fixture(self, assets_dir):
        return {
            "custom_models": {
                "enabled": False,
            },
            "self_signed_jwt": {
                "signing_key": open(assets_dir / "keys" / "signing_key.pem").read(),
            },
        }

    @pytest.fixture(name="scopes")
    def scopes_fixture(self):
        return ["complete_code", "generate_code"]

    @pytest.fixture(name="route")
    def route_fixture(self):
        return "/code/suggestions"

    def test_suggestions_ssrf_model_metadata_endpoint_rejected_when_custom_models_disabled(
        self,
        mock_client: TestClient,
        route: str,
    ):
        with pytest.raises(
            ValueError, match="specifying custom models endpoint is disabled"
        ):
            mock_client.post(
                route,
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-GitLab-Instance-Id": "1234",
                    "X-GitLab-Realm": "self-managed",
                    "X-Gitlab-Global-User-Id": "test-user-id",
                },
                json={
                    "prompt_components": [
                        {
                            "type": "code_editor_completion",
                            "payload": {
                                "file_name": "main.py",
                                "content_above_cursor": "# test",
                                "content_below_cursor": "\n",
                                "language_identifier": "python",
                            },
                        }
                    ],
                    "model_metadata": {
                        "endpoint": "http://internal-server.local/ssrf",
                        "api_key": "malicious-key",
                        "identifier": "provider/model",
                        "name": "mistral",
                        "provider": "litellm",
                    },
                },
            )
