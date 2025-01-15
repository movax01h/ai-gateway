# pylint: disable=too-many-lines

import time
from typing import Dict, List, Union
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
from ai_gateway.config import Config, ConfigModelEndpoints, ConfigModelKeys
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
            scopes=["complete_code", "generate_code"],
            subject="1234",
            gitlab_realm="self-managed",
            issuer="issuer",
        ),
    )


@pytest.fixture
def mock_config():
    config = Config()
    config.custom_models.enabled = True
    config.model_keys = ConfigModelKeys(
        fireworks_api_key="mock_fireworks_key",
    )
    config.model_endpoints = ConfigModelEndpoints(
        fireworks_current_region_endpoint={
            "endpoint": "https://fireworks.endpoint",
            "identifier": "qwen2p5-coder-7b",
        }
    )

    yield config


@pytest.fixture
def unit_primitives():
    return ["complete_code", "generate_code"]


class TestCodeCompletions:
    def cleanup(self):
        """Ensure Snowplow cache is reset between tests."""
        yield
        Snowplow.reset()

    @pytest.mark.parametrize(
        ("mock_completions_legacy_output_texts", "expected_response"),
        [
            # non-empty suggestions from model
            (
                ["def search", "println"],
                {
                    "id": "id",
                    "model": {
                        "engine": "vertex-ai",
                        "name": "code-gecko",
                        "lang": "python",
                        "tokens_consumption_metadata": {
                            "input_tokens": 1,
                            "output_tokens": 2,
                            "context_tokens_sent": None,
                            "context_tokens_used": None,
                        },
                    },
                    "experiments": [{"name": "truncate_suffix", "variant": 1}],
                    "object": "text_completion",
                    "enabled_feature_flags": ["flag_a", "flag_b"],
                    "created": 1695182638,
                    "choices": [
                        {
                            "text": "def search",
                            "index": 0,
                            "finish_reason": "length",
                        },
                        {
                            "text": "println",
                            "index": 0,
                            "finish_reason": "length",
                        },
                    ],
                },
            ),
            # empty suggestions from model
            (
                [""],
                {
                    "id": "id",
                    "model": {
                        "engine": "vertex-ai",
                        "name": "code-gecko",
                        "lang": "python",
                        "tokens_consumption_metadata": None,
                    },
                    "experiments": [{"name": "truncate_suffix", "variant": 1}],
                    "enabled_feature_flags": ["flag_a", "flag_b"],
                    "object": "text_completion",
                    "created": 1695182638,
                    "choices": [],
                },
            ),
        ],
    )
    def test_legacy_successful_response(
        self,
        mock_client: TestClient,
        mock_track_internal_event,
        mock_completions_legacy: Mock,
        expected_response: dict,
    ):
        current_feature_flag_context.set({"flag_a", "flag_b"})

        response = mock_client.post(
            "/completions",
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
                "context": [
                    {"content": "test context", "name": "test", "type": "function"}
                ],
            },
        )

        assert response.status_code == 200
        mock_completions_legacy.assert_called_once_with(
            prefix="# Create a fast binary search\n",
            suffix="\n",
            file_name="main.py",
            editor_lang=None,
            stream=False,
            snowplow_event_context=ANY,
        )

        body = response.json()

        assert body["choices"] == expected_response["choices"]
        assert body["experiments"] == expected_response["experiments"]
        assert body["model"] == expected_response["model"]
        assert set(body["metadata"]["enabled_feature_flags"]) == set(
            expected_response["enabled_feature_flags"]
        )

        mock_track_internal_event.assert_called_once_with(
            "request_complete_code",
            category="ai_gateway.api.v2.code.completions",
        )

    @pytest.mark.parametrize(
        ("headers", "expected_args"),
        [
            # Omitted Language Server Version:
            (
                {},
                {},
            ),
            # Supported Language Server Version:
            (
                {"X-Gitlab-Language-Server-Version": "6.3.0"},
                {"code_context": ["import numpy as np"]},
            ),
            # Unsupported Language Server Version:
            (
                {"X-Gitlab-Language-Server-Version": "4.15.0"},
                {},
            ),
        ],
    )
    def test_completions_legacy_advanced_context_support(
        self,
        mock_client: TestClient,
        mock_completions_legacy: Mock,
        headers: dict,
        expected_args: dict,
    ):
        current_file = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
        }
        response = mock_client.post(
            "/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
                **headers,
            },
            json={
                "prompt_version": 1,
                "project_path": "gitlab-org/gitlab",
                "project_id": 278964,
                "current_file": current_file,
                "context": [
                    {
                        "type": "file",
                        "name": "other.py",
                        "content": "import numpy as np",
                    }
                ],
            },
        )
        assert response.status_code == 200
        mock_completions_legacy.assert_called_with(
            prefix=current_file["content_above_cursor"],
            suffix=current_file["content_below_cursor"],
            file_name=current_file["file_name"],
            editor_lang=current_file.get("language_identifier", None),
            stream=False,
            snowplow_event_context=ANY,
            **expected_args,
        )

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

        code_completions_kwargs = (
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
            raw_prompt = [
                Message(role=prompt["role"], content=prompt["content"])
                for prompt in raw_prompt
            ]
            code_completions_kwargs.update({"raw_prompt": raw_prompt})

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

        if want_status == 200:
            body = response.json()
            assert body["choices"] == want_choices

        if model_provider == "fireworks_ai":
            mock_llm_text.assert_called_with(
                "foo",
                "\n",
                False,
                snowplow_event_context=ANY,
                max_output_tokens=64,
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
        mock_completions_legacy: Mock,
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

    def _send_code_completions_request(self, mock_client, params):
        headers = {
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
            "X-GitLab-Instance-Id": "1234",
            "X-GitLab-Realm": "self-managed",
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
                "claude-2.0",
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
                "claude-2.0",
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
                "claude-3-opus-20240229",
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

    @pytest.mark.asyncio
    @capture_validation_errors()
    async def test_generations_with_validation_error(self, mock_client):
        response = mock_client.post(
            "/code/generations",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "current_file": {
                    "file_name": "main.py",
                    "content_above_cursor": "# create function",
                    "content_below_cursor": "\n",
                },
                "prompt_version": 2,
                "prompt": "# create a function",
                "model_provider": "anthropic",
                "model_name": "claude-2.1",
                "stream": True,
                "choices_count": 1,
                "prompt_id": "12345",
            },
        )

        assert response.status_code == 422

        body = response.json()
        assert "endpoint" in body["detail"]
        assert "URL input should be a string or URL" in body["detail"]

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
