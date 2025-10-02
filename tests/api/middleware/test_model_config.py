import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware import Middleware

from ai_gateway.api.middleware.model_config import ModelConfigMiddleware
from ai_gateway.model_metadata import current_model_metadata_context


@pytest.fixture(name="model_metadata_context")
def model_metadata_context_fixture():
    current_model_metadata_context.set(None)
    yield current_model_metadata_context


@pytest.fixture(name="middleware_test_client")
def middleware_test_client_fixture(model_metadata_context):
    app = FastAPI(middleware=[Middleware(ModelConfigMiddleware)])

    @app.post("/test")
    async def do_something(
        request: Request,
    ):
        await request.body()
        model_metadata = model_metadata_context.get()
        return (
            model_metadata.model_dump(
                exclude={"llm_definition_params", "family", "friendly_name"}
            )
            if model_metadata
            else None
        )

    return TestClient(app)


@pytest.mark.asyncio
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
async def test_handles_large_request_body_with_model_metadata(middleware_test_client):
    """Test that model_metadata is extracted from large request bodies (chunked requests)"""
    model_params = {
        "name": "claude_3",
        "provider": "openai",
        "endpoint": "http://bedrockselfhostedmodel.com/",
        "api_key": "",
        "identifier": "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
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


@pytest.mark.asyncio
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
