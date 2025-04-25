import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware import Middleware

from ai_gateway.api.middleware.model_config import ModelConfigMiddleware
from ai_gateway.model_metadata import current_model_metadata_context


@pytest.fixture
def model_metadata_context():
    current_model_metadata_context.set(None)
    yield current_model_metadata_context


@pytest.fixture
def middleware_test_client(model_metadata_context):
    app = FastAPI(middleware=[Middleware(ModelConfigMiddleware)])

    @app.post("/test")
    async def do_something(
        request: Request,
    ):
        await request.body()
        return model_metadata_context.get()

    return TestClient(app)


@pytest.mark.asyncio
async def test_parses_model_params_into_context_var(middleware_test_client):
    model_params = {
        "name": "test_model",
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
