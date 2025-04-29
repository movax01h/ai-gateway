# pylint: disable=too-many-lines

import time
from typing import Any, Dict, List, Union
from unittest.mock import ANY, Mock, patch

import pytest
from dependency_injector import containers
from fastapi import status
from fastapi.testclient import TestClient
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from snowplow_tracker import Snowplow
from starlette.datastructures import CommaSeparatedStrings
from structlog.testing import capture_logs

from ai_gateway.api.error_utils import capture_validation_errors
from ai_gateway.api.v2 import api_router
from ai_gateway.feature_flags.context import current_feature_flag_context
from ai_gateway.models.base_chat import Message, Role
from ai_gateway.tracking.container import ContainerTracking
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator
from ai_gateway.tracking.snowplow import SnowplowEvent, SnowplowEventContext


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def auth_user():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            scopes=["complete_code", "generate_code", "amazon_q_integration"],
            subject="1234",
            gitlab_realm="self-managed",
            issuer="issuer",
        ),
    )


@pytest.fixture
def config_values(assets_dir, gcp_location):
    return {
        "custom_models": {
            "enabled": True,
        },
        "google_cloud_platform": {
            "location": gcp_location,
        },
        "model_keys": {
            "fireworks_api_key": "mock_fireworks_key",
        },
        "self_signed_jwt": {
            "signing_key": open(assets_dir / "keys" / "signing_key.pem").read(),
        },
        "amazon_q": {
            "region": "us-west-2",
        },
        "model_endpoints": {
            "fireworks_regional_endpoints": {
                gcp_location: {
                    "qwen2p5-coder-7b": {
                        "endpoint": "https://fireworks.endpoint",
                        "identifier": "qwen2p5-coder-7b",
                    },
                    "codestral-2501": {
                        "endpoint": "https://fireworks.endpoint",
                        "identifier": "codestral-2501",
                    },
                },
            },
        },
    }


@pytest.fixture
def unit_primitives():
    return ["complete_code", "generate_code"]


@pytest.fixture
def gcp_location():
    return "us-central1"


@pytest.fixture
def mock_post_processor():
    with patch("ai_gateway.code_suggestions.completions.PostProcessor.process") as mock:
        mock.return_value = "Post-processed completion response"

        yield mock


class TestCodeCompletions:
    def cleanup(self):
        """Ensure Snowplow cache is reset between tests."""
        yield
        Snowplow.reset()

    @pytest.mark.parametrize(
        (
            "prompt_version",
            "mock_suggestions_engine",
            "mock_suggestions_model",
            "mock_suggestions_output_text",
            "expected_response",
            "expect_context",
        ),
        [
            # non-empty suggestions from model
            (
                1,
                "anthropic",
                "claude-2.1",
                "def search",
                {
                    "id": "id",
                    "model": {
                        "engine": "anthropic",
                        "name": "claude-2.1",
                        "lang": "python",
                        "tokens_consumption_metadata": None,
                        "region": "us-central1",
                    },
                    "object": "text_completion",
                    "created": 1695182638,
                    "choices": [
                        {
                            "text": "def search",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ],
                    "enabled_feature_flags": ["flag_b", "flag_c"],
                },
                False,
            ),
            # prompt version 3
            (
                3,
                "anthropic",
                "claude-3-5-sonnet-20240620",
                "def search",
                {
                    "id": "id",
                    "model": {
                        "engine": "anthropic",
                        "name": "claude-3-5-sonnet-20240620",
                        "lang": "python",
                        "tokens_consumption_metadata": None,
                        "region": "us-central1",
                    },
                    "object": "text_completion",
                    "created": 1695182638,
                    "choices": [
                        {
                            "text": "def search",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ],
                    "enabled_feature_flags": ["flag_b", "flag_c"],
                },
                False,
            ),
            # codestral
            (
                2,
                "codestral",
                "codestral",
                "def search",
                {
                    "id": "id",
                    "model": {
                        "engine": "codestral",
                        "name": "codestral",
                        "lang": "python",
                        "tokens_consumption_metadata": None,
                        "region": "us-central1",
                    },
                    "object": "text_completion",
                    "created": 1695182638,
                    "choices": [
                        {
                            "text": "def search",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ],
                    "enabled_feature_flags": ["flag_b", "flag_c"],
                },
                True,
            ),
            # empty suggestions from model
            (
                1,
                "anthropic",
                "claude-2.1",
                "",
                {
                    "id": "id",
                    "model": {
                        "engine": "anthropic",
                        "name": "claude-2.1",
                        "lang": "python",
                        "tokens_consumption_metadata": None,
                        "region": "us-central1",
                    },
                    "object": "text_completion",
                    "created": 1695182638,
                    "choices": [],
                    "enabled_feature_flags": ["flag_b", "flag_c"],
                },
                False,
            ),
        ],
    )
    def test_successful_response(
        self,
        prompt_version: int,
        mock_suggestions_engine: str,
        mock_suggestions_model: str,
        mock_client: TestClient,
        mock_completions: Mock,
        expected_response: dict,
        expect_context: bool,
    ):
        current_file = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
        }
        data = {
            "prompt_version": prompt_version,
            "project_path": "gitlab-org/gitlab",
            "project_id": 278964,
            "model_provider": mock_suggestions_engine,
            "model_name": mock_suggestions_model,
            "current_file": current_file,
            "context": [
                {"content": "test context", "name": "test", "type": "function"}
            ],
        }

        code_completions_kwargs: dict[str, Any] = (
            {"code_context": ["test context"]} if expect_context else {}
        )
        if prompt_version == 2:
            data.update(
                {
                    "prompt": current_file["content_above_cursor"],
                }
            )

        if prompt_version == 3 and mock_suggestions_engine == "anthropic":
            raw_prompt = [
                {
                    "role": "system",
                    "content": "You are a code completion tool that performs Fill-in-the-middle. ",
                },
                {
                    "role": "user",
                    "content": "<SUFFIX>\n// write a function to find the max\n</SUFFIX>\n<PREFIX>\n\n\treturn min\n}\n</PREFIX>')]",
                },
            ]
            data.update({"prompt": raw_prompt})

            raw_prompt_args = [
                Message(role=Role(prompt["role"]), content=prompt["content"])
                for prompt in raw_prompt
            ]
            code_completions_kwargs.update({"raw_prompt": raw_prompt_args})

        current_feature_flag_context.set({"flag_b", "flag_c"})

        response = mock_client.post(
            "/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json=data,
        )

        assert response.status_code == 200

        body = response.json()

        assert body["choices"] == expected_response["choices"]
        assert body["model"] == expected_response["model"]
        assert set(body["metadata"]["enabled_feature_flags"]) == set(
            expected_response["enabled_feature_flags"]
        )

        mock_completions.assert_called_with(
            prefix=current_file["content_above_cursor"],
            suffix=current_file["content_below_cursor"],
            file_name=current_file["file_name"],
            editor_lang=current_file.get("language_identifier", None),
            stream=False,
            snowplow_event_context=ANY,
            **code_completions_kwargs,
        )

    @pytest.mark.parametrize("prompt_version", [1])
    @pytest.mark.xdist_group("capture_logs")
    def test_request_latency(
        self,
        prompt_version: int,
        mock_client: TestClient,
        mock_completions: Mock,
    ):
        current_file = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
        }
        data = {
            "prompt_version": prompt_version,
            "project_path": "gitlab-org/gitlab",
            "project_id": 278964,
            "model_provider": "anthropic",
            "current_file": current_file,
            "model_name": "claude-3-5-sonnet-20240620",
        }

        code_completions_kwargs = {}
        if prompt_version == 2:
            data.update(
                {
                    "prompt": current_file["content_above_cursor"],
                }
            )
            code_completions_kwargs.update(
                {"raw_prompt": current_file["content_above_cursor"]}
            )

        def get_request_duration(cap_logs):
            event = 'testclient:50000 - "POST /completions HTTP/1.1" 200'
            entry = next(entry for entry in cap_logs if entry["event"] == event)

            return entry["duration_request"]

        # get valid duration
        with capture_logs() as cap_logs:
            response = mock_client.post(
                "/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Rails-Send-Start": str(time.time() - 1),
                    "X-GitLab-Instance-Id": "1234",
                    "X-GitLab-Realm": "self-managed",
                },
                json=data,
            )
            assert response.status_code == 200
            assert get_request_duration(cap_logs) >= 1

        # -1 in duration_request indicates invalid X-Gitlab-Rails-Send-Start header
        with capture_logs() as cap_logs:
            response = mock_client.post(
                "/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Rails-Send-Start": "invalid epoch time",
                    "X-GitLab-Instance-Id": "1234",
                    "X-GitLab-Realm": "self-managed",
                },
                json=data,
            )
            assert response.status_code == 200
            assert get_request_duration(cap_logs) == -1

        # -1 in duration_request indicates missing X-Gitlab-Rails-Send-Start header
        with capture_logs() as cap_logs:
            response = mock_client.post(
                "/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-GitLab-Instance-Id": "1234",
                    "X-GitLab-Realm": "self-managed",
                },
                json=data,
            )
            assert response.status_code == 200
            assert get_request_duration(cap_logs) == -1

    @pytest.mark.parametrize(
        (
            "prompt_version",
            "prompt",
            "context",
            "model_provider",
            "model_name",
            "model_endpoint",
            "model_api_key",
            "model_identifier",
            "want_litellm_called",
            "want_prompt_called",
            "want_amazon_q_called",
            "want_status",
            "want_choices",
        ),
        [
            # test litellm completions
            (
                2,
                "foo",
                [],
                "litellm",
                "codegemma",
                "http://localhost:4000/",
                "api-key",
                None,
                True,
                False,
                False,
                200,
                [
                    {
                        "text": "test completion",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
            ),
            (
                2,
                "foo",
                [],
                "litellm",
                "codegemma",
                "http://localhost:4000/",
                "api-key",
                "provider/codegemma_2b",
                True,
                False,
                False,
                200,
                [
                    {
                        "text": "test completion",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
            ),
            (
                2,
                "",
                [],
                "litellm",
                "codegemma",
                "http://localhost:4000/",
                "api-key",
                None,
                False,
                True,
                False,
                200,
                [
                    {
                        "text": "test completion",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
            ),
            (
                2,
                None,
                [],
                "litellm",
                "codegemma",
                "http://localhost:4000/",
                "api-key",
                None,
                False,
                True,
                False,
                200,
                [
                    {
                        "text": "test completion",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
            ),
            (
                2,
                "foo",
                [],
                "codestral",
                "codestral",
                "http://localhost:4000/",
                "api-key",
                None,
                True,
                False,
                False,
                200,
                [
                    {
                        "text": "test completion",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
            ),
            (
                2,
                "foo",
                [],
                "fireworks_ai",
                "qwen2p5-coder-7b",
                "https://fireworks.endpoint",
                "api-key",
                None,
                True,
                False,
                False,
                200,
                [
                    {
                        "text": "test completion",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
            ),
            (
                2,
                "foo",
                [
                    {
                        "type": "file",
                        "name": "other.py",
                        "content": "import numpy as np",
                    },
                    {
                        "type": "file",
                        "name": "bar.py",
                        "content": "from typing import Any",
                    },
                ],
                "fireworks_ai",
                "qwen2p5-coder-7b",
                "https://fireworks.endpoint",
                "api-key",
                None,
                True,
                False,
                False,
                200,
                [
                    {
                        "text": "test completion",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
            ),
            (
                2,
                None,
                [],
                "amazon_q",
                "amazon_q",
                None,
                None,
                None,
                False,
                False,
                True,
                200,
                [
                    {
                        "text": "test completion",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
            ),
            (
                1,
                None,
                [],
                "vertex-ai",
                "codestral-2501",
                None,
                None,
                None,
                True,
                False,
                False,
                200,
                [
                    {
                        "text": "test completion",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
            ),
            (
                2,
                None,
                [],
                "vertex-ai",
                "codestral-2501",
                None,
                None,
                None,
                True,
                False,
                False,
                200,
                [
                    {
                        "text": "test completion",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
            ),
        ],
    )
    def test_non_stream_response(
        self,
        mock_client,
        mock_llm_text: Mock,
        mock_agent_model: Mock,
        mock_amazon_q_model: Mock,
        mock_track_internal_event: Mock,
        mock_config,
        prompt_version,
        prompt,
        context,
        model_provider,
        model_name,
        model_endpoint,
        model_api_key,
        model_identifier,
        want_litellm_called,
        want_prompt_called,
        want_amazon_q_called,
        want_status,
        want_choices,
    ):
        response = mock_client.post(
            "/code/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
                "X-GitLab-Global-User-Id": "1234",
            },
            json={
                "prompt_version": prompt_version,
                "project_path": "gitlab-org/gitlab",
                "project_id": 278964,
                "current_file": {
                    "file_name": "main.py",
                    "content_above_cursor": "foo",
                    "content_below_cursor": "\n",
                },
                "prompt": prompt,
                "context": context,
                "model_provider": model_provider,
                "model_name": model_name,
                "model_endpoint": model_endpoint,
                "model_api_key": model_api_key,
                "model_identifier": model_identifier,
            },
        )

        assert response.status_code == want_status
        assert mock_llm_text.called == want_litellm_called
        assert mock_agent_model.called == want_prompt_called
        assert mock_amazon_q_model.called == want_amazon_q_called

        if want_status == 200:
            body = response.json()
            assert body["choices"] == want_choices

        if model_provider == "fireworks_ai":
            content = [c["content"] for c in context]
            prefix = "\n".join(content + ["foo"])

            mock_llm_text.assert_called_with(
                prefix,
                "\n",
                False,
                snowplow_event_context=ANY,
                max_output_tokens=48,
            )

        if model_provider == "amazon_q":
            mock_track_internal_event.assert_called_once_with(
                "request_amazon_q_integration_complete_code",
                category="ai_gateway.api.v2.code.completions",
            )

    @pytest.mark.parametrize(
        (
            "expected_response",
            "provider",
            "model",
            "extra_args",
            "context",
        ),
        [
            (
                "def search",
                "anthropic",
                None,
                {},
                [],
            ),
            (
                "def search",
                "litellm",
                "codestral",
                {},
                [],
            ),
            (
                "def search",
                "vertex-ai",
                "codestral-2501",
                {
                    "temperature": 0.7,
                    "max_output_tokens": 64,
                    "context_max_percent": 0.3,
                },
                [],
            ),
            (
                "def search",
                "vertex-ai",
                "codestral-2501",
                {
                    "temperature": 0.7,
                    "max_output_tokens": 64,
                    "context_max_percent": 0.3,
                },
                [{"name": "test", "type": "file", "content": "some context"}],
            ),
        ],
    )
    def test_successful_stream_response(
        self,
        mock_client: TestClient,
        mock_completions_stream: Mock,
        expected_response: str,
        provider: str,
        extra_args: dict,
        context: list,
        model: str | None,
    ):
        current_file = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
        }
        data = {
            "prompt_version": 1,
            "project_path": "gitlab-org/gitlab",
            "project_id": 278964,
            "model_provider": provider,
            "current_file": current_file,
            "stream": True,
            "context": context,
            "model_name": "claude-3-5-sonnet-20240620",
        }
        if model:
            data["model_name"] = model

        response = mock_client.post(
            "/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json=data,
        )

        assert response.status_code == 200
        assert response.text == expected_response
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        completions_args = {
            "prefix": current_file["content_above_cursor"],
            "suffix": current_file["content_below_cursor"],
            "file_name": current_file["file_name"],
            "editor_lang": current_file.get("language_identifier", None),
            "stream": True,
            "snowplow_event_context": ANY,
        }

        if context:
            completions_args["code_context"] = [c["content"] for c in context]

        completions_args.update(extra_args)

        mock_completions_stream.assert_called_with(**completions_args)

    @pytest.mark.parametrize(
        (
            "auth_user",
            "telemetry",
            "current_file",
            "request_headers",
            "expected_language",
        ),
        [
            (
                CloudConnectorUser(
                    authenticated=True,
                    claims=UserClaims(
                        scopes=["complete_code"],
                        subject="1234",
                        gitlab_realm="self-managed",
                        issuer="gitlab-ai-gateway",
                    ),
                ),
                [
                    {
                        "model_engine": "vertex",
                        "model_name": "code-gecko",
                        "requests": 1,
                        "accepts": 1,
                        "errors": 0,
                        "lang": None,
                    }
                ],
                {
                    "file_name": "main.py",
                    "content_above_cursor": "# Create a fast binary search\n",
                    "content_below_cursor": "\n",
                },
                {
                    "User-Agent": "vs-code",
                    "X-Gitlab-Instance-Id": "1234",
                    "X-Gitlab-Global-User-Id": "1234",
                    "X-Gitlab-Host-Name": "gitlab.com",
                    "X-Gitlab-Saas-Duo-Pro-Namespace-Ids": "4,5,6",
                    "X-Gitlab-Realm": "self-managed",
                },
                "python",
            )
        ],
    )
    def test_snowplow_tracking(
        self,
        mock_client: TestClient,
        mock_container: containers.Container,
        mock_completions: Mock,
        auth_user: CloudConnectorUser,
        telemetry: List[Dict[str, Union[str, int, None]]],
        current_file: Dict[str, str],
        expected_language: str,
        request_headers: Dict[str, str],
    ):
        expected_event = SnowplowEvent(
            context=SnowplowEventContext(
                prefix_length=len(current_file.get("content_above_cursor", "")),
                suffix_length=len(current_file.get("content_below_cursor", "")),
                language=expected_language,
                gitlab_realm=request_headers.get("X-Gitlab-Realm", ""),
                is_direct_connection=True,
                suggestion_source="network",
                region="us-central1",
                gitlab_instance_id=request_headers.get("X-Gitlab-Instance-Id", ""),
                gitlab_global_user_id=request_headers.get(
                    "X-Gitlab-Global-User-Id", ""
                ),
                gitlab_host_name=request_headers.get("X-Gitlab-Host-Name", ""),
                gitlab_saas_duo_pro_namespace_ids=list(
                    map(
                        int,
                        CommaSeparatedStrings(
                            request_headers.get(
                                "X-Gitlab-Saas-Duo-Pro-Namespace-Ids", ""
                            )
                        ),
                    )
                ),
            )
        )

        snowplow_instrumentator_mock = Mock(spec=SnowplowInstrumentator)

        snowplow_container_mock = Mock(spec=ContainerTracking)
        snowplow_container_mock.instrumentator = Mock(
            return_value=snowplow_instrumentator_mock
        )

        with patch.object(mock_container, "snowplow", snowplow_container_mock):
            mock_client.post(
                "/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-GitLab-Instance-Id": "1234",
                    "X-GitLab-Realm": "self-managed",
                    **request_headers,
                },
                json={
                    "prompt_version": 1,
                    "project_path": "gitlab-org/gitlab",
                    "project_id": 278964,
                    "current_file": current_file,
                    "telemetry": telemetry,
                    "choices_count": 1,
                },
            )

        snowplow_instrumentator_mock.watch.assert_called_once()
        args = snowplow_instrumentator_mock.watch.call_args[0]
        assert len(args) == 1
        assert args[0] == expected_event

    @pytest.mark.parametrize(
        ("auth_user", "extra_headers", "expected_status_code"),
        [
            (
                CloudConnectorUser(
                    authenticated=True,
                    claims=UserClaims(
                        scopes=["complete_code"],
                        subject="1234",
                        gitlab_realm="self-managed",
                    ),
                ),
                {"X-GitLab-Instance-Id": "1234"},
                200,
            ),
            (
                CloudConnectorUser(
                    authenticated=True,
                    claims=UserClaims(
                        scopes=["complete_code"],
                        subject="1234",
                        gitlab_realm="self-managed",
                        issuer="gitlab-ai-gateway",
                    ),
                ),
                {"X-Gitlab-Global-User-Id": "1234"},
                200,
            ),
        ],
    )
    def test_successful_response_with_correct_issuers(
        self,
        mock_client: TestClient,
        mock_completions: Mock,
        auth_user: CloudConnectorUser,
        extra_headers: Dict[str, str],
        expected_status_code: int,
    ):
        current_file = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
        }
        data = {
            "prompt_version": 1,
            "project_path": "gitlab-org/gitlab",
            "project_id": 278964,
            "model_provider": "anthropic",
            "current_file": current_file,
            "model_name": "claude-3-5-sonnet-20240620",
        }

        response = mock_client.post(
            "/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Realm": "self-managed",
            }
            | extra_headers,
            json=data,
        )

        assert response.status_code == expected_status_code

    @pytest.mark.parametrize(
        "model_details",
        [{"model_provider": "vertex-ai", "model_name": "codestral-2501"}, {}],
    )
    def test_vertex_codestral(
        self,
        mock_client: Mock,
        mock_litellm_acompletion: Mock,
        mock_post_processor: Mock,
        model_details: Dict[str, str],
    ):
        params = {
            "prompt_version": 2,
            "project_path": "gitlab-org/gitlab",
            "project_id": 278964,
            "current_file": {
                "file_name": "main.py",
                "content_above_cursor": "foo",
                "content_below_cursor": "\n",
            },
        }

        params.update(model_details)

        response = self._send_code_completions_request(mock_client, params)

        mock_litellm_acompletion.assert_called_with(
            model="vertex_ai/codestral-2501",
            messages=[{"content": "foo", "role": Role.USER}],
            suffix="\n",
            text_completion=True,
            vertex_ai_location="us-central1",
            max_tokens=64,
            temperature=0.7,
            top_p=0.95,
            stream=False,
            timeout=60,
            stop=["\n\n", "\n+++++", "[PREFIX]", "</s>[SUFFIX]", "[MIDDLE]"],
        )

        mock_post_processor.assert_called_with(
            "Test text completion response",
            score=10**5,
            max_output_tokens_used=False,
            model_name="vertex_ai/codestral-2501",
        )

        result = response.json()
        assert result["model"]["engine"] == "vertex-ai"
        assert result["model"]["name"] == "vertex_ai/codestral-2501"
        assert result["choices"][0]["text"] == "Post-processed completion response"

    def test_vertex_codestral_with_prompt(self, mock_client, mock_agent_model: Mock):
        params = {
            "prompt_version": 2,
            "project_path": "gitlab-org/gitlab",
            "project_id": 278964,
            "current_file": {
                "file_name": "main.py",
                "content_above_cursor": "foo",
                "content_below_cursor": "\n",
            },
            "prompt": "bar",
            "model_provider": "vertex-ai",
            "model_name": "codestral-2501",
        }

        response = self._send_code_completions_request(mock_client, params)

        assert not mock_agent_model.called
        assert response.status_code == 400

        body = response.json()
        assert (
            (body["detail"])
            == "You cannot specify a prompt with the given provider and model combination"
        )

    @pytest.mark.parametrize(
        "allow_llm_cache",
        ["true", "false"],
    )
    def test_fireworks_codestral_with_prompt(
        self, allow_llm_cache: str, mock_litellm_acompletion: Mock, mock_client: Mock
    ):
        params = {
            "project_path": "gitlab-org/gitlab-shell",
            "project_id": 278964,
            "current_file": {
                "file_name": "main.py",
                "content_above_cursor": "foo",
                "content_below_cursor": "\n",
            },
            "model_provider": "fireworks_ai",
            "model_name": "codestral-2501",
            "stream": False,
            "choices_count": 1,
            "prompt_version": 1,
            "prompt_cache_max_len": 0,
        }

        self._send_code_completions_request(mock_client, params, allow_llm_cache)

        kwargs = mock_litellm_acompletion.call_args.kwargs

        if allow_llm_cache == "false":
            assert "prompt_cache_max_len" in kwargs
        else:
            assert "prompt_cache_max_len" not in kwargs

    @pytest.mark.parametrize(
        ("gcp_location", "model_details", "expected_model", "expected_content"),
        [
            (
                "asia-mock-location",
                {"model_provider": "vertex-ai", "model_name": "codestral-2501"},
                "codestral-2501",
                "</s>[SUFFIX]\n[PREFIX]foo[MIDDLE]",
            ),
            (
                "asia-mock-location",
                {},
                "qwen2p5-coder-7b",
                "<|fim_prefix|>foo<|fim_suffix|>\n<|fim_middle|>",
            ),
        ],
    )
    def test_code_completion_in_asia(
        self,
        mock_client: Mock,
        mock_litellm_acompletion: Mock,
        model_details: Dict[str, str],
        expected_model: str,
        expected_content: str,
    ):
        params = {
            "prompt_version": 1,
            "project_path": "gitlab-org/gitlab",
            "project_id": 278964,
            "current_file": {
                "file_name": "main.py",
                "content_above_cursor": "foo",
                "content_below_cursor": "\n",
            },
        }

        params.update(model_details)

        self._send_code_completions_request(mock_client, params)

        mock_litellm_acompletion.assert_called_once_with(
            messages=[{"content": expected_content, "role": "user"}],
            max_tokens=48,
            temperature=0.95,
            top_p=0.95,
            timeout=60,
            stop=ANY,
            api_base="https://fireworks.endpoint",
            api_key="mock_fireworks_key",
            model=expected_model,
            custom_llm_provider="text-completion-openai",
            client=None,
            prompt_cache_max_len=0,
            logprobs=1,
            stream=False,
        )

    @pytest.mark.asyncio
    @capture_validation_errors()
    async def test_completions_with_validation_error(self, mock_client):
        params = {
            "current_file": {
                "file_name": "main.py",
                "content_above_cursor": "foo",
                "language_identifier": "python",
                "content_below_cursor": "}",
            },
            "prompt_version": 2,
            "model_provider": "codestral",
            "model_name": "dummy_model",
        }

        response = self._send_code_completions_request(mock_client, params)

        assert response.status_code == 422

        body = response.json()
        expected_error_message = [
            {
                "ctx": {"error": {}},
                "input": "dummy_model",
                "loc": ["body", 2, "model_name"],
                "msg": "Value error, model dummy_model is not supported by use case code completions and provider codestral",
                "type": "value_error",
            }
        ]

        assert (body["detail"]) == expected_error_message

    def test_disable_code_gecko_default(
        self,
        mock_client: Mock,
        mock_litellm_acompletion: Mock,
        mock_post_processor: Mock,
    ):
        params = {
            "prompt_version": 1,
            "project_path": "gitlab-org/gitlab",
            "project_id": 278964,
            "current_file": {
                "file_name": "main.py",
                "content_above_cursor": "foo",
                "content_below_cursor": "\n",
            },
        }
        response = self._send_code_completions_request(mock_client, params)

        mock_litellm_acompletion.assert_called_with(
            model="vertex_ai/codestral-2501",
            messages=[{"content": "foo", "role": Role.USER}],
            suffix="\n",
            text_completion=True,
            vertex_ai_location="us-central1",
            max_tokens=64,
            temperature=0.7,
            top_p=0.95,
            stream=False,
            timeout=60,
            stop=["\n\n", "\n+++++", "[PREFIX]", "</s>[SUFFIX]", "[MIDDLE]"],
        )

        mock_post_processor.assert_called_with(
            "Test text completion response",
            score=10**5,
            max_output_tokens_used=False,
            model_name="vertex_ai/codestral-2501",
        )

        result = response.json()
        assert result["model"]["engine"] == "vertex-ai"
        assert result["model"]["name"] == "vertex_ai/codestral-2501"
        assert result["choices"][0]["text"] == "Post-processed completion response"

    @pytest.mark.parametrize("gcp_location", ["asia-mock-location"])
    def test_disable_code_gecko_default_in_asia(
        self,
        mock_client: Mock,
        mock_litellm_acompletion: Mock,
        mock_post_processor: Mock,
    ):
        params = {
            "prompt_version": 1,
            "project_path": "gitlab-org/gitlab",
            "project_id": 278964,
            "current_file": {
                "file_name": "main.py",
                "content_above_cursor": "foo",
                "content_below_cursor": "\n",
            },
        }
        response = self._send_code_completions_request(mock_client, params)

        mock_litellm_acompletion.assert_called_with(
            messages=[
                {
                    "content": "<|fim_prefix|>foo<|fim_suffix|>\n<|fim_middle|>",
                    "role": Role.USER,
                }
            ],
            max_tokens=48,
            temperature=0.95,
            top_p=0.95,
            stream=False,
            timeout=60,
            stop=[
                "<|fim_prefix|>",
                "<|fim_suffix|>",
                "<|fim_middle|>",
                "<|fim_pad|>",
                "<|repo_name|>",
                "<|file_sep|>",
                "<|im_start|>",
                "<|im_end|>",
                "\n\n",
            ],
            api_base="https://fireworks.endpoint",
            api_key="mock_fireworks_key",
            model="qwen2p5-coder-7b",
            custom_llm_provider="text-completion-openai",
            prompt_cache_max_len=0,
            logprobs=1,
            client=ANY,
        )

        mock_post_processor.assert_called_with(
            "Test response",
            score=999,
            max_output_tokens_used=False,
            model_name="text-completion-fireworks_ai/qwen2p5-coder-7b",
        )

        result = response.json()
        assert result["model"]["engine"] == "fireworks_ai"
        assert (
            result["model"]["name"] == "text-completion-fireworks_ai/qwen2p5-coder-7b"
        )
        assert result["choices"][0]["text"] == "Post-processed completion response"

    def _send_code_completions_request(
        self, mock_client, params, enable_prompt_cache="false"
    ):
        headers = {
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
            "X-GitLab-Instance-Id": "1234",
            "X-GitLab-Realm": "self-managed",
            "X-Gitlab-Model-Prompt-Cache-Enabled": enable_prompt_cache,
        }

        return mock_client.post(
            "/code/completions",
            headers=headers,
            json=params,
        )


class TestCodeGenerations:
    def cleanup(self):
        """Ensure Snowplow cache is reset between tests."""
        yield
        Snowplow.reset()

    @pytest.mark.parametrize(
        (
            "prompt_version",
            "prompt_id",
            "prefix",
            "prompt",
            "model_provider",
            "model_name",
            "model_endpoint",
            "model_api_key",
            "mock_output_text",
            "want_vertex_called",
            "want_anthropic_called",
            "want_anthropic_chat_called",
            "want_prompt_prepared_called",
            "want_litellm_called",
            "want_status",
            "want_prompt",
            "want_choices",
            "enabled_feature_flags",
        ),
        [
            (
                1,
                None,
                "foo",
                None,
                "vertex-ai",
                "code-bison@002",
                None,
                None,
                "foo",
                True,
                False,
                False,
                False,
                False,
                200,
                None,
                [
                    {
                        "text": "\nfoo",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
                ["flag_a", "flag_b"],
            ),  # v1 without prompt - vertex-ai
            (
                1,
                None,
                "foo",
                None,
                "anthropic",
                "claude-2.1",
                None,
                None,
                "foo",
                False,
                True,
                False,
                False,
                False,
                200,
                None,
                [
                    {
                        "text": "foo",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
                ["flag_a", "flag_b"],
            ),  # v1 without prompt - anthropic
            (
                1,
                None,
                "foo",
                "bar",
                "vertex-ai",
                "code-bison@002",
                None,
                None,
                "foo",
                True,
                False,
                False,
                False,
                False,
                200,
                None,
                [
                    {
                        "text": "\nfoo",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
                ["flag_a", "flag_b"],
            ),  # v1 with prompt - vertex-ai
            (
                1,
                None,
                "foo",
                "bar",
                "vertex-ai",
                "code-bison@002",
                None,
                None,
                "foo",
                True,
                False,
                False,
                False,
                False,
                200,
                None,
                [
                    {
                        "text": "\nfoo",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
                ["flag_a", "flag_b"],
            ),  # v1 with prompt - anthropic
            (
                1,
                None,
                "foo",
                "bar",
                "vertex-ai",
                "code-bison@002",
                None,
                None,
                "",
                True,
                False,
                False,
                False,
                False,
                200,
                None,
                [],
                ["flag_a", "flag_b"],
            ),  # v1 empty suggestions from model
            (
                2,
                None,
                "foo",
                "bar",
                "vertex-ai",
                "code-bison@002",
                None,
                None,
                "foo",
                True,
                False,
                False,
                True,
                False,
                200,
                "bar",
                [
                    {
                        "text": "\nfoo",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
                ["flag_a", "flag_b"],
            ),  # v2 with prompt - vertex-ai
            (
                2,
                None,
                "foo",
                "bar",
                "anthropic",
                "claude-2.1",
                None,
                None,
                "foo",
                False,
                True,
                False,
                True,
                False,
                200,
                "bar",
                [
                    {
                        "text": "foo",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
                ["flag_a", "flag_b"],
            ),  # v2 with prompt - anthropic
            (
                2,
                None,
                "foo",
                None,
                "anthropic",
                "claude-2.1",
                None,
                None,
                "foo",
                False,
                False,
                False,
                False,
                False,
                422,
                None,
                None,
                ["flag_a", "flag_b"],
            ),  # v2 without prompt field
            (
                2,
                None,
                "foo",
                "bar",
                "anthropic",
                "claude-2.1",
                None,
                None,
                "",
                False,
                True,
                False,
                True,
                False,
                200,
                "bar",
                [],
                ["flag_a", "flag_b"],
            ),  # v2 empty suggestions from model
            (
                3,
                None,
                "foo",
                [
                    {"role": "system", "content": "foo"},
                    {"role": "user", "content": "bar"},
                ],
                "anthropic",
                "claude-3-5-haiku-20241022",
                None,
                None,
                "foo",
                False,
                False,
                True,
                True,
                False,
                200,
                [
                    Message(role=Role.SYSTEM, content="foo"),
                    Message(role=Role.USER, content="bar"),
                ],
                [
                    {
                        "text": "foo",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
                ["flag_a", "flag_b"],
            ),  # v3 with prompt - anthropic
            (
                3,
                None,
                "foo",
                [
                    {"role": "system", "content": "foo"},
                    {"role": "user", "content": "bar"},
                ],
                "anthropic",
                "claude-3-5-haiku-20241022",
                None,
                None,
                "foo",
                False,
                False,
                True,
                True,
                False,
                200,
                [
                    Message(role=Role.SYSTEM, content="foo"),
                    Message(role=Role.USER, content="bar"),
                ],
                [
                    {
                        "text": "foo",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
                ["flag_a", "flag_b"],
            ),
            (
                3,
                "",
                "foo",
                [
                    {"role": "system", "content": "foo"},
                    {"role": "user", "content": "bar"},
                ],
                "litellm",
                "mistral",
                "http://localhost:11434/v1",
                "api-key",
                "foo",
                False,
                False,
                False,
                False,
                True,
                200,
                [
                    Message(role=Role.SYSTEM, content="foo"),
                    Message(role=Role.USER, content="bar"),
                ],
                [
                    {
                        "text": "\nfoo",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
                ["flag_a", "flag_b"],
            ),  # v3 with prompt - litellm
            (
                2,
                "code_suggestions/generations",
                "foo",
                "prompt",
                "litellm",
                "mistral",
                "http://localhost:11434/v1",
                "api-key",
                "foo",
                False,
                False,
                False,
                False,
                False,
                200,
                [
                    Message(role=Role.SYSTEM, content="foo"),
                    Message(role=Role.USER, content="bar"),
                ],
                [
                    {
                        "text": "\nfoo",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
                ["flag_a", "flag_b"],
            ),
        ],
    )
    def test_non_stream_response(
        self,
        mock_client,
        mock_container: containers.Container,
        mock_code_bison: Mock,
        mock_anthropic: Mock,
        mock_anthropic_chat: Mock,
        mock_llm_chat: Mock,
        mock_agent_model: Mock,
        mock_with_prompt_prepared: Mock,
        prompt_version,
        prompt_id,
        prefix,
        prompt,
        model_provider,
        model_name,
        model_endpoint,
        model_api_key,
        mock_output_text,
        want_vertex_called,
        want_anthropic_called,
        want_anthropic_chat_called,
        want_prompt_prepared_called,
        want_litellm_called,
        want_status,
        want_prompt,
        want_choices,
        enabled_feature_flags,
    ):
        current_feature_flag_context.set({"flag_a", "flag_b"})

        response = mock_client.post(
            "/code/generations",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_version": prompt_version,
                "project_path": "gitlab-org/gitlab",
                "project_id": 278964,
                "current_file": {
                    "file_name": "main.py",
                    "content_above_cursor": prefix,
                    "content_below_cursor": "\n",
                },
                "prompt": prompt,
                "model_provider": model_provider,
                "model_name": model_name,
                "model_endpoint": model_endpoint,
                "model_api_key": model_api_key,
                "prompt_id": prompt_id,
            },
        )

        assert response.status_code == want_status
        assert mock_code_bison.called == want_vertex_called
        assert mock_anthropic.called == want_anthropic_called
        assert mock_llm_chat.called == want_litellm_called
        assert mock_anthropic_chat.called == want_anthropic_chat_called
        assert mock_agent_model.called == bool(prompt_id)

        if want_prompt_prepared_called:
            mock_with_prompt_prepared.assert_called_with(want_prompt)

        if want_status == 200:
            body = response.json()
            assert body["choices"] == want_choices
            assert set(body["metadata"]["enabled_feature_flags"]) == set(
                enabled_feature_flags
            )

    def test_successful_stream_response(
        self,
        mock_client: TestClient,
        mock_generations_stream: Mock,
        mock_suggestions_output_text: str,
    ):
        response = mock_client.post(
            "/code/generations",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_version": 2,
                "project_path": "gitlab-org/gitlab",
                "project_id": 278964,
                "current_file": {
                    "file_name": "main.py",
                    "content_above_cursor": "# create function",
                    "content_below_cursor": "\n",
                },
                "prompt": "# create a function",
                "model_provider": "anthropic",
            },
        )

        assert response.status_code == 200
        assert response.text == mock_suggestions_output_text
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    @pytest.mark.parametrize(
        (
            "telemetry",
            "current_file",
            "request_headers",
            "expected_language",
        ),
        [
            (
                [
                    {
                        "model_engine": "vertex",
                        "model_name": "code-gecko",
                        "requests": 1,
                        "accepts": 1,
                        "errors": 0,
                        "lang": None,
                    }
                ],
                {
                    "file_name": "main.py",
                    "content_above_cursor": "# Create a fast binary search\n",
                    "content_below_cursor": "\n",
                },
                {
                    "User-Agent": "vs-code",
                    "X-Gitlab-Instance-Id": "1234",
                    "X-Gitlab-Global-User-Id": "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
                    "X-Gitlab-Host-Name": "gitlab.com",
                    "X-Gitlab-Saas-Duo-Pro-Namespace-Ids": "4,5,6",
                    "X-Gitlab-Realm": "self-managed",
                },
                "python",
            )
        ],
    )
    def test_snowplow_tracking(
        self,
        mock_client: TestClient,
        mock_container: containers.Container,
        mock_generations: Mock,
        telemetry: List[Dict[str, Union[str, int, None]]],
        current_file: Dict[str, str],
        expected_language: str,
        request_headers: Dict[str, str],
    ):
        expected_event = SnowplowEvent(
            context=SnowplowEventContext(
                prefix_length=len(current_file.get("content_above_cursor", "")),
                suffix_length=len(current_file.get("content_below_cursor", "")),
                language=expected_language,
                gitlab_realm=request_headers.get("X-Gitlab-Realm", ""),
                is_direct_connection=False,
                suggestion_source="network",
                region="us-central1",
                gitlab_instance_id=request_headers.get("X-Gitlab-Instance-Id", ""),
                gitlab_global_user_id=request_headers.get(
                    "X-Gitlab-Global-User-Id", ""
                ),
                gitlab_host_name=request_headers.get("X-Gitlab-Host-Name", ""),
                gitlab_saas_duo_pro_namespace_ids=list(
                    map(
                        int,
                        CommaSeparatedStrings(
                            request_headers.get(
                                "X-Gitlab-Saas-Duo-Pro-Namespace-Ids", ""
                            )
                        ),
                    )
                ),
            )
        )

        snowplow_instrumentator_mock = Mock(spec=SnowplowInstrumentator)

        snowplow_container_mock = Mock(spec=ContainerTracking)
        snowplow_container_mock.instrumentator = Mock(
            return_value=snowplow_instrumentator_mock
        )

        with patch.object(mock_container, "snowplow", snowplow_container_mock):
            mock_client.post(
                "/code/generations",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    **request_headers,
                },
                json={
                    "prompt_version": 1,
                    "project_path": "gitlab-org/gitlab",
                    "project_id": 278964,
                    "current_file": current_file,
                    "prompt": "some prompt",
                    "model_provider": "vertex-ai",
                    "model_name": "code-bison@002",
                    "telemetry": telemetry,
                    "choices_count": 1,
                },
            )

        snowplow_instrumentator_mock.watch.assert_called_once()
        args = snowplow_instrumentator_mock.watch.call_args[0]
        assert len(args) == 1
        assert args[0] == expected_event


class TestUnauthorizedScopes:
    @pytest.fixture
    def auth_user(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                scopes=["unauthorized_scope"],
                subject="1234",
                gitlab_realm="self-managed",
            ),
        )

    @pytest.mark.parametrize(
        ("path", "error_message"),
        [
            ("/completions", "Unauthorized to access code completions"),
            ("/code/generations", "Unauthorized to access code generations"),
        ],
    )
    def test_failed_authorization_scope(self, mock_client, path, error_message):
        response = mock_client.post(
            path,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_version": 1,
                "project_path": "gitlab-org/gitlab",
                "project_id": 278964,
                "current_file": {
                    "file_name": "main.py",
                    "content_above_cursor": "# Create a fast binary search\n",
                    "content_below_cursor": "\n",
                },
            },
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert response.json() == {"detail": error_message}


class TestUnauthorizedIssuer:
    @pytest.fixture
    def auth_user(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                scopes=["complete_code", "generate_code"],
                subject="1234",
                gitlab_realm="self-managed",
                issuer="gitlab-ai-gateway",
            ),
        )

    def test_failed_authorization_scope(self, mock_client):
        response = mock_client.post(
            "/code/generations",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_version": 1,
                "project_path": "gitlab-org/gitlab",
                "project_id": 278964,
                "current_file": {
                    "file_name": "main.py",
                    "content_above_cursor": "# Create a fast binary search\n",
                    "content_below_cursor": "\n",
                },
            },
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert response.json() == {"detail": "Unauthorized to access code generations"}


class TestAmazonQIntegration:
    @pytest.fixture
    def auth_user(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                scopes=["amazon_q_integration"],
                subject="1234",
                gitlab_realm="self-managed",
                issuer="gitlab-ai-gateway",
            ),
        )

    @pytest.mark.parametrize(
        ("auth_user", "model_provider", "model_name"),
        [
            (
                CloudConnectorUser(
                    authenticated=True,
                    claims=UserClaims(
                        scopes=["amazon_q_integration"],
                        subject="1234",
                        gitlab_realm="self-managed",
                        issuer="gitlab-ai-gateway",
                    ),
                ),
                "anthropic",
                "claude-3-5-sonnet-20240620",
            ),
            (
                CloudConnectorUser(
                    authenticated=True,
                    claims=UserClaims(
                        scopes=["complete_code", "generate_code"],
                        subject="1234",
                        gitlab_realm="self-managed",
                        issuer="gitlab-ai-gateway",
                    ),
                ),
                "amazon_q",
                "amazon_q",
            ),
        ],
    )
    def test_failed_authorization_scope(
        self,
        mock_client,
        auth_user,
        model_provider,
        model_name,
    ):
        response = mock_client.post(
            "/code/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
                "X-GitLab-Global-User-Id": "1234",
            },
            json={
                "prompt_version": 2,
                "project_path": "gitlab-org/gitlab",
                "project_id": 278964,
                "current_file": {
                    "file_name": "main.py",
                    "content_above_cursor": "foo",
                    "content_below_cursor": "\n",
                },
                "model_provider": model_provider,
                "model_name": model_name,
                "role_arn": "role-arn",
            },
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert response.json() == {"detail": "Unauthorized to access code completions"}
