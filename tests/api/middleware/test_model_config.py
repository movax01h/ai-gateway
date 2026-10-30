import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from gitlab_cloud_connector import CloudConnectorUser
from starlette.middleware import Middleware
from starlette.types import ASGIApp, Receive, Scope, Send
from structlog.testing import capture_logs

from ai_gateway.api.middleware.model_config import ModelConfigMiddleware
from ai_gateway.model_selection import ModelSelectionConfig, UnitPrimitiveConfig
from ai_gateway.prompts.feature_roots import discover_feature_prompts
from ai_gateway.prompts.registry import PROMPT_ID_TO_FEATURE_SETTING
from lib.context import StarletteUser, current_model_metadata_context


@pytest.fixture(name="model_metadata_context")
def model_metadata_context_fixture():
    current_model_metadata_context.set(None)
    yield current_model_metadata_context


@pytest.fixture(name="middleware_test_client")
def middleware_test_client_fixture(model_metadata_context):
    app = FastAPI(
        middleware=[Middleware(ModelConfigMiddleware, skip_endpoints=["/skipped"])]
    )

    @app.get("/body")
    @app.post("/body")
    @app.post("/skipped")
    async def body(request: Request):
        return {
            "body": await request.json(),
            "model_metadata_set": model_metadata_context.get() is not None,
        }

    @app.post("/test")
    async def do_something(
        request: Request,
    ):
        await request.body()
        model_metadata = model_metadata_context.get()
        return (
            model_metadata.model_dump(
                exclude={
                    "llm_definition",
                    "friendly_name",
                    "iam_role",
                }
            )
            if model_metadata
            else None
        )

    @app.post("/v1/embeddings/code_embeddings/{operation}")
    async def code_embeddings(request: Request, operation: str):
        await request.body()
        model_metadata = model_metadata_context.get()
        return {
            "operation": operation,
            "name": model_metadata.name if model_metadata else None,
        }

    @app.post("/v1/prompts/{prompt_id:path}")
    @app.post("/v2/prompts/{prompt_id:path}")
    async def prompts(request: Request, prompt_id: str):
        await request.body()
        model_metadata = model_metadata_context.get()
        return {
            "prompt_id": prompt_id,
            "name": model_metadata.name if model_metadata else None,
        }

    @app.post("/iam-role")
    async def iam_role(request: Request):
        await request.body()
        model_metadata = model_metadata_context.get()
        return {
            "iam_role": model_metadata.iam_role if model_metadata else None,
        }

    return TestClient(app)


@pytest.mark.asyncio
@pytest.mark.usefixtures("custom_models_gateway")
async def test_parses_model_params_into_context_var(middleware_test_client):
    model_params = {
        "name": "gpt",
        "provider": "custom_openai",
        "endpoint": "http://test_model.com/",
        "api_key": "test_api_key",
        "identifier": "test_model_identifier",
    }

    response = middleware_test_client.post(
        "/test", json={"model_metadata": model_params}
    )

    assert response.json() == model_params


@pytest.mark.usefixtures("custom_models_gateway")
@pytest.mark.parametrize(
    ("method", "path", "model_metadata_set"),
    [
        pytest.param("POST", "/body", True, id="post"),
        pytest.param("GET", "/body", False, id="get"),
        pytest.param("POST", "/skipped", False, id="skipped-endpoint"),
    ],
)
def test_reads_the_body_only_for_requests_that_can_use_model_metadata(
    middleware_test_client, method, path, model_metadata_set
):
    payload = {
        "model_metadata": {
            "name": "gpt",
            "provider": "custom_openai",
            "endpoint": "http://test_model.com/",
            "api_key": "test_api_key",
            "identifier": "test_model_identifier",
        }
    }

    response = middleware_test_client.request(method, path, json=payload)

    assert response.json() == {
        "body": payload,
        "model_metadata_set": model_metadata_set,
    }


@pytest.mark.asyncio
async def test_amazon_q_params_to_context_var(middleware_test_client):
    model_params = {
        "provider": "amazon_q",
        "name": "amazon_q",
        "role_arn": "arn:aws:iam::123456789012:role/example-role",
    }

    response = middleware_test_client.post(
        "/test", json={"model_metadata": model_params}
    )

    assert response.json() == model_params


@pytest.mark.asyncio
async def test_model_params_is_none_when_body_is_empty(middleware_test_client):
    response = middleware_test_client.post("/test")

    assert response.json() is None


@pytest.mark.asyncio
async def test_model_params_is_none_when_metadata_is_not_in_body(
    middleware_test_client,
):
    response = middleware_test_client.post("/test", json={"foo": "bar"})

    assert response.json() is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("custom_models_gateway")
async def test_handles_large_request_body_with_model_metadata(middleware_test_client):
    """Test that model_metadata is extracted from large request bodies (chunked requests)"""
    model_params = {
        "name": "claude_3",
        "provider": "openai",
        "endpoint": "http://bedrockselfhostedmodel.com/",
        "api_key": "",
        "identifier": "bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    }

    # Create a large payload to simulate chunked request
    large_content = "x" * 10000  # 10KB of content
    large_payload = {
        "model_metadata": model_params,
        "inputs": {
            "content": large_content,
            "file_content": large_content,
            "additional_data": large_content,
        },
    }

    response = middleware_test_client.post("/test", json=large_payload)

    assert response.json() == model_params


@pytest.mark.asyncio
@pytest.mark.usefixtures("custom_models_gateway")
async def test_handles_very_large_request_body(middleware_test_client):
    """Test with very large request body (>1MB) to ensure chunking is handled."""
    model_params = {
        "name": "gpt",
        "provider": "custom_openai",
        "endpoint": "http://test_model.com/",
        "api_key": "test_key",
        "identifier": "test_identifier",
    }

    # Create a very large payload (>1MB)
    very_large_content = "x" * (1024 * 1024 + 1000)  # ~1MB + 1000 bytes
    large_payload = {
        "model_metadata": model_params,
        "inputs": {"file_content": very_large_content},
    }

    response = middleware_test_client.post("/test", json=large_payload)

    assert response.json() == model_params


@pytest.mark.asyncio
async def test_handles_malformed_json_in_large_body(middleware_test_client):
    """Test that malformed JSON in large bodies doesn't crash the middleware."""
    # Send malformed JSON as raw data
    malformed_json = (
        '{"model_metadata": {"name": "test"}, "invalid": json}' + "x" * 5000
    )

    response = middleware_test_client.post(
        "/test", data=malformed_json, headers={"content-type": "application/json"}
    )

    # Should not crash and return None for model_metadata
    assert response.json() is None


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b'["model_metadata"]', id="array-body"),
        pytest.param(b'"model_metadata"', id="string-body"),
        pytest.param(b'{"model_metadata": 5}', id="number-value"),
        pytest.param(b'{"model_metadata": true}', id="boolean-value"),
        pytest.param(b'{"model_metadata": "provider"}', id="string-value"),
        pytest.param(b'{"model_metadata": ["provider"]}', id="array-value"),
    ],
)
def test_ignores_model_metadata_that_is_not_an_object(middleware_test_client, body):
    response = middleware_test_client.post(
        "/test", content=body, headers={"content-type": "application/json"}
    )

    assert response.status_code == 200
    assert response.json() is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("custom_models_gateway")
async def test_handles_unicode_in_large_body(middleware_test_client):
    """Test that unicode characters in large bodies are handled correctly."""
    model_params = {
        "name": "gpt",
        "provider": "custom_openai",
        "endpoint": "http://test_model.com/",
        "api_key": "test_key",
        "identifier": "test_identifier",
    }

    # Create payload with unicode characters
    unicode_content = "测试内容 🚀 " * 1000  # Unicode content repeated
    payload = {"model_metadata": model_params, "inputs": {"content": unicode_content}}

    response = middleware_test_client.post("/test", json=payload)

    assert response.json() == model_params


@pytest.mark.asyncio
@pytest.mark.usefixtures("custom_models_gateway")
async def test_model_metadata_at_end_of_large_body(middleware_test_client):
    """Test that model_metadata is found even when it appears at the end of a large body."""
    model_params = {
        "name": "gpt",
        "provider": "custom_openai",
        "endpoint": "http://test_model.com/",
        "api_key": "test_key",
        "identifier": "test_identifier",
    }

    # Put model_metadata at the end of a large payload
    large_content = "x" * 8000
    payload = {
        "inputs": {
            "large_field_1": large_content,
            "large_field_2": large_content,
            "large_field_3": large_content,
        },
        "model_metadata": model_params,  # At the end
    }

    response = middleware_test_client.post("/test", json=payload)

    assert response.json() == model_params


@pytest.mark.asyncio
async def test_empty_model_metadata_in_large_body(middleware_test_client):
    """Test handling of empty model_metadata in large bodies."""
    large_content = "x" * 5000
    payload = {
        "model_metadata": {},  # Empty model_metadata
        "inputs": {"content": large_content},
    }

    response = middleware_test_client.post("/test", json=payload)

    assert response.json() is None


@pytest.mark.asyncio
async def test_null_model_metadata_in_large_body(middleware_test_client):
    """Test handling of null model_metadata in large bodies."""
    large_content = "x" * 5000
    payload = {
        "model_metadata": None,  # Null model_metadata
        "inputs": {"content": large_content},
    }

    response = middleware_test_client.post("/test", json=payload)

    # Should return None when model_metadata is explicitly null
    assert response.json() is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("custom_models_gateway")
async def test_multiple_model_metadata_fields_in_large_body(middleware_test_client):
    """Test that only the first model_metadata field is processed in large bodies."""
    model_params_1 = {
        "name": "gpt",
        "provider": "provider_1",
        "endpoint": "http://model1.com/",
        "api_key": "key_1",
        "identifier": "id_1",
    }

    model_params_2 = {
        "name": "mistral",
        "provider": "provider_2",
        "endpoint": "http://model2.com/",
        "api_key": "key_2",
        "identifier": "id_2",
    }

    large_content = "x" * 3000
    payload = {
        "model_metadata": model_params_1,  # First occurrence
        "inputs": {
            "content": large_content,
            "model_metadata": model_params_2,  # Second occurrence (should be ignored)
        },
    }

    response = middleware_test_client.post("/test", json=payload)

    # Should use the first model_metadata found
    assert response.json() == model_params_1


BEDROCK_ENDPOINT = "https://bedrock-runtime.us-east-1.amazonaws.com/"
IAM_ROLE_ARN = "arn:aws:iam::123456789012:role/my-role"


@pytest.mark.asyncio
@pytest.mark.usefixtures("cloud_gateway")
async def test_rejects_iam_role_when_custom_models_disabled(middleware_test_client):
    response = middleware_test_client.post(
        "/iam-role",
        json={
            "model_metadata": {
                "name": "gpt",
                "provider": "bedrock",
                "endpoint": BEDROCK_ENDPOINT,
                "identifier": "bedrock/test_model_identifier",
                "iam_role": IAM_ROLE_ARN,
            }
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "iam_role is only supported when custom models are enabled"
    }


@pytest.mark.asyncio
@pytest.mark.usefixtures("custom_models_gateway")
async def test_parses_iam_role_into_context_var(middleware_test_client):
    model_params = {
        "name": "gpt",
        "provider": "bedrock",
        "endpoint": BEDROCK_ENDPOINT,
        "identifier": "bedrock/test_model_identifier",
        "iam_role": IAM_ROLE_ARN,
    }

    response = middleware_test_client.post(
        "/iam-role", json={"model_metadata": model_params}
    )

    assert response.json() == {"iam_role": IAM_ROLE_ARN}


FABLE = "claude_fable_5"
CHAT = "duo_chat"
NOT_ALLOWED_FOR_CHAT = {
    "provider": "gitlab",
    "identifier": FABLE,
    "feature_setting": CHAT,
}


CODE_COMPLETIONS_MODEL = {
    "provider": "gitlab",
    "identifier": "codestral_2508_fireworks",
    "feature_setting": "code_completions",
}


def _allowlist_logs(logs, event):
    return [
        (
            entry["reason"],
            entry["feature_setting"],
            entry.get("prompt_id"),
            entry["path"],
        )
        for entry in logs
        if entry["event"] == event
    ]


def _feature(feature_setting: str) -> UnitPrimitiveConfig:
    return ModelSelectionConfig.instance().get_unit_primitive_config_map()[
        feature_setting
    ]


class _AuthenticatedUser:
    """Set the user the way the authentication middleware does."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope["user"] = StarletteUser(CloudConnectorUser(True, global_user_id="user-1"))
        await self.app(scope, receive, send)


@pytest.mark.usefixtures("cloud_gateway", "feature_allowlist_mode")
class TestFeatureAllowlist:
    @pytest.mark.parametrize(
        "feature_allowlist_mode", ["log", "fallback", "enforce"], indirect=True
    )
    def test_serves_an_allowed_model(self, middleware_test_client):
        allowed = _feature(CHAT).default_model_identifiers[0]

        response = middleware_test_client.post(
            "/test",
            json={
                "model_metadata": {
                    "provider": "gitlab",
                    "identifier": allowed,
                    "feature_setting": CHAT,
                }
            },
        )

        assert response.status_code == 200
        assert response.json()["name"] == allowed

    @pytest.mark.parametrize("feature_allowlist_mode", ["log"], indirect=True)
    def test_log_mode_serves_the_model_and_logs_the_mismatch(
        self, middleware_test_client
    ):
        assert FABLE not in _feature(CHAT).allowed_model_identifiers

        with capture_logs() as logs:
            response = middleware_test_client.post(
                "/test", json={"model_metadata": NOT_ALLOWED_FOR_CHAT}
            )

        assert response.json()["name"] == FABLE
        assert [
            (entry["reason"], entry["path"])
            for entry in logs
            if entry["event"] == "Model allowlist mismatch"
        ] == [("model_not_allowed", "/test")]

    @pytest.mark.parametrize("feature_allowlist_mode", ["fallback"], indirect=True)
    def test_fallback_mode_serves_the_feature_default(self, middleware_test_client):
        response = middleware_test_client.post(
            "/test", json={"model_metadata": NOT_ALLOWED_FOR_CHAT}
        )

        assert response.json()["name"] in _feature(CHAT).default_model_identifiers

    @pytest.mark.parametrize("feature_allowlist_mode", ["fallback"], indirect=True)
    def test_fallback_mode_leaves_a_model_without_a_feature_on_the_prompt_default(
        self, middleware_test_client
    ):
        response = middleware_test_client.post(
            "/test", json={"model_metadata": {"provider": "gitlab", "name": FABLE}}
        )

        assert response.json() is None

    @pytest.mark.parametrize("feature_allowlist_mode", ["fallback"], indirect=True)
    def test_a_fireworks_default_gets_the_user_session(self, model_metadata_context):
        app = FastAPI(
            middleware=[
                Middleware(_AuthenticatedUser),
                Middleware(ModelConfigMiddleware),
            ]
        )

        @app.post("/test")
        async def session_id(request: Request):
            await request.body()
            return {"session_id": model_metadata_context.get().session_id}

        response = TestClient(app).post(
            "/test",
            json={
                "model_metadata": {
                    "provider": "gitlab",
                    "identifier": FABLE,
                    "feature_setting": "code_completions",
                }
            },
        )

        assert response.json() == {"session_id": "user-1"}

    @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
    @pytest.mark.parametrize(
        ("model_metadata", "detail"),
        [
            pytest.param(
                NOT_ALLOWED_FOR_CHAT,
                "Model 'claude_fable_5' is not available for 'duo_chat'",
                id="model-not-allowed",
            ),
            pytest.param(
                {"provider": "openai", "name": FABLE},
                "This gateway does not serve models from provider 'openai'",
                id="provider-not-served",
            ),
            pytest.param(
                {"provider": "gitlab", "name": FABLE},
                "feature_setting is required to request model 'claude_fable_5'",
                id="missing-feature-setting",
            ),
        ],
    )
    def test_enforce_mode_rejects_the_request(
        self, middleware_test_client, model_metadata, detail
    ):
        response = middleware_test_client.post(
            "/test", json={"model_metadata": model_metadata}
        )

        assert response.status_code == 403
        assert response.json() == {"detail": detail}

    class TestCodeEmbeddingsRoutes:
        """GitLab Rails sends code embeddings requests without a `feature_setting`."""

        RAILS_REQUEST = {
            "provider": "gitlab",
            "identifier": "text_embedding_005_vertex",
        }
        OPERATIONS = ["index", "search"]

        @pytest.mark.parametrize("feature_allowlist_mode", ["log"], indirect=True)
        @pytest.mark.parametrize("operation", OPERATIONS)
        def test_log_mode_logs_no_mismatch(self, middleware_test_client, operation):
            with capture_logs() as logs:
                response = middleware_test_client.post(
                    f"/v1/embeddings/code_embeddings/{operation}",
                    json={"model_metadata": self.RAILS_REQUEST},
                )

            assert response.json()["name"] == "text_embedding_005_vertex"
            assert [
                entry for entry in logs if entry["event"] == "Model allowlist mismatch"
            ] == []

        @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
        @pytest.mark.parametrize("operation", OPERATIONS)
        def test_enforce_mode_serves_the_embeddings_model(
            self, middleware_test_client, operation
        ):
            response = middleware_test_client.post(
                f"/v1/embeddings/code_embeddings/{operation}",
                json={"model_metadata": self.RAILS_REQUEST},
            )

            assert response.status_code == 200
            assert response.json() == {
                "operation": operation,
                "name": "text_embedding_005_vertex",
            }

        @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
        def test_enforce_mode_rejects_a_model_of_another_feature(
            self, middleware_test_client
        ):
            response = middleware_test_client.post(
                "/v1/embeddings/code_embeddings/index",
                json={
                    "model_metadata": {
                        "provider": "gitlab",
                        "identifier": "claude_sonnet_4_6_vertex",
                    }
                },
            )

            assert response.status_code == 403
            assert response.json() == {
                "detail": "Model 'claude_sonnet_4_6_vertex' is not available for 'embeddings_code'"
            }

    class TestPromptRoutes:
        """A prompt in `PROMPT_ID_TO_FEATURE_SETTING` is checked against its feature; any other prompt gets a note."""

        class TestMappedPrompts:
            PATHS = ["/v1/prompts/chat/explain_code", "/v2/prompts/chat/explain_code"]

            @pytest.mark.parametrize("feature_allowlist_mode", ["log"], indirect=True)
            @pytest.mark.parametrize("path", PATHS)
            def test_log_mode_checks_the_feature_rails_sends(
                self, middleware_test_client, path
            ):
                assert (
                    "codestral_2508_fireworks"
                    not in _feature("duo_chat_explain_code").allowed_model_identifiers
                )

                with capture_logs() as logs:
                    response = middleware_test_client.post(
                        path,
                        json={"model_metadata": CODE_COMPLETIONS_MODEL},
                    )

                assert response.json()["name"] == "codestral_2508_fireworks"
                assert _allowlist_logs(logs, "Model allowlist mismatch") == [
                    (
                        "model_not_allowed",
                        "duo_chat_explain_code",
                        "chat/explain_code",
                        path,
                    )
                ]
                assert _allowlist_logs(logs, "Model allowlist note") == []

            @pytest.mark.parametrize(
                "feature_allowlist_mode", ["fallback"], indirect=True
            )
            @pytest.mark.parametrize("path", PATHS)
            def test_fallback_mode_serves_the_default_of_the_feature_rails_sends(
                self, middleware_test_client, path
            ):
                # The client's feature allows the model, so only the mapped feature replaces it.
                assert (
                    "codestral_2508_fireworks"
                    in _feature("code_completions").allowed_model_identifiers
                )

                response = middleware_test_client.post(
                    path,
                    json={"model_metadata": CODE_COMPLETIONS_MODEL},
                )

                assert (
                    response.json()["name"]
                    == (_feature("duo_chat_explain_code").default_model_identifiers[0])
                )

            @pytest.mark.parametrize(
                "feature_allowlist_mode", ["enforce"], indirect=True
            )
            @pytest.mark.parametrize("path", PATHS)
            def test_enforce_mode_rejects_a_model_of_another_feature(
                self, middleware_test_client, path
            ):
                response = middleware_test_client.post(
                    path,
                    json={"model_metadata": CODE_COMPLETIONS_MODEL},
                )

                assert response.status_code == 403
                assert response.json() == {
                    "detail": "Model 'codestral_2508_fireworks' is not available for 'duo_chat_explain_code'"
                }

            @pytest.mark.parametrize(
                "feature_allowlist_mode", ["enforce"], indirect=True
            )
            @pytest.mark.parametrize(
                ("path", "feature_setting"),
                [
                    ("/v1/prompts/chat/explain_code/", "duo_chat_explain_code"),
                    ("/v1/prompts/chat//explain_code", "duo_chat_explain_code"),
                    (
                        "/v1/prompts/review/review_merge_request",
                        "review_merge_request",
                    ),
                ],
            )
            def test_enforce_mode_checks_another_spelling_of_the_prompt_id(
                self, middleware_test_client, path, feature_setting
            ):
                discover_feature_prompts()

                response = middleware_test_client.post(
                    path,
                    json={"model_metadata": CODE_COMPLETIONS_MODEL},
                )

                assert response.status_code == 403
                assert response.json() == {
                    "detail": f"Model 'codestral_2508_fireworks' is not available for '{feature_setting}'"
                }

            @pytest.mark.parametrize(
                "feature_allowlist_mode", ["enforce"], indirect=True
            )
            @pytest.mark.parametrize(
                ("prompt_id", "feature_setting"),
                sorted(PROMPT_ID_TO_FEATURE_SETTING.items()),
            )
            def test_enforce_mode_serves_what_rails_sends(
                self, middleware_test_client, prompt_id, feature_setting
            ):
                default_model = _feature(feature_setting).default_model_identifiers[0]

                with capture_logs() as logs:
                    response = middleware_test_client.post(
                        f"/v1/prompts/{prompt_id}",
                        json={
                            "model_metadata": {
                                "provider": "gitlab",
                                "identifier": default_model,
                                "feature_setting": feature_setting,
                            }
                        },
                    )

                assert response.status_code == 200
                assert response.json()["name"] == default_model
                assert [
                    entry
                    for entry in logs
                    if entry["event"].startswith("Model allowlist")
                ] == []

            @pytest.mark.parametrize(
                "feature_allowlist_mode", ["enforce"], indirect=True
            )
            def test_categorize_question_runs_on_the_chat_model(
                self, middleware_test_client
            ):
                assert (
                    "claude_haiku_4_5_20251001"
                    not in _feature("categorize_question").allowed_model_identifiers
                )

                with capture_logs() as logs:
                    response = middleware_test_client.post(
                        "/v1/prompts/categorize_question",
                        json={
                            "model_metadata": {
                                "provider": "gitlab",
                                "identifier": "claude_haiku_4_5_20251001",
                                "feature_setting": "duo_chat",
                            }
                        },
                    )

                assert response.status_code == 200
                assert [
                    entry
                    for entry in logs
                    if entry["event"].startswith("Model allowlist")
                ] == []

        class TestOtherPrompts:
            PATHS = [
                "/v1/prompts/generate_description",
                "/v2/prompts/generate_description",
            ]

            @pytest.mark.parametrize(
                "feature_allowlist_mode", ["log", "fallback", "enforce"], indirect=True
            )
            @pytest.mark.parametrize("path", PATHS)
            def test_notes_the_model_and_serves_it(self, middleware_test_client, path):
                assert (
                    "codestral_2508_fireworks"
                    not in _feature("generate_description").allowed_model_identifiers
                )

                with capture_logs() as logs:
                    response = middleware_test_client.post(
                        path,
                        json={"model_metadata": CODE_COMPLETIONS_MODEL},
                    )

                assert response.status_code == 200
                assert response.json() == {
                    "prompt_id": "generate_description",
                    "name": "codestral_2508_fireworks",
                }
                assert _allowlist_logs(logs, "Model allowlist note") == [
                    (
                        "prompt_feature_model_not_allowed",
                        "generate_description",
                        "generate_description",
                        path,
                    )
                ]

            @pytest.mark.parametrize(
                ("path", "model_metadata"),
                [
                    pytest.param(
                        PATHS[0],
                        {
                            "provider": "gitlab",
                            "identifier": "codestral_2508_fireworks",
                            "feature_setting": "generate_description",
                        },
                        id="client-sends-the-prompt-feature",
                    ),
                    pytest.param(
                        "/test",
                        {
                            "provider": "gitlab",
                            "identifier": "codestral_2508_fireworks",
                            "feature_setting": "code_completions",
                        },
                        id="not-a-prompt-route",
                    ),
                ],
            )
            def test_notes_nothing(self, middleware_test_client, path, model_metadata):
                with capture_logs() as logs:
                    middleware_test_client.post(
                        path, json={"model_metadata": model_metadata}
                    )

                assert _allowlist_logs(logs, "Model allowlist note") == []
