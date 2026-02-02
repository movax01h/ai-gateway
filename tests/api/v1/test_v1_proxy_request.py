import json

import pytest
from fastapi import HTTPException, Response
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.api.v1.proxy.request import (
    track_billing_event,
    verify_project_namespace_metadata,
)
from ai_gateway.proxy.clients.token_usage import TokenUsage
from lib.billing_events import BillingEvent
from lib.internal_events.context import EventContext, current_event_context


@pytest.fixture
def setup_token_usage(mock_request):  # pylint: disable=redefined-outer-name
    def _setup(
        input_tokens=0,
        output_tokens=0,
        total_tokens=None,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        reasoning_tokens=0,
        model_name=None,
    ):
        mock_request.state.proxy_token_usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            reasoning_tokens=reasoning_tokens,
        )
        if model_name:
            mock_request.state.proxy_model_name = model_name

    return _setup


@pytest.mark.asyncio
async def test_track_billing_event(mock_request, billing_event_client):
    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        return Response(content=b'{"message": "success"}', status_code=200)

    await dummy_func(mock_request, billing_event_client=billing_event_client)

    billing_event_client.track_billing_event.assert_called_once_with(
        mock_request.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category="ai_gateway.api.v1.proxy.request",
        unit_of_measure="request",
        quantity=1,
        metadata=None,
    )


@pytest.mark.asyncio
async def test_track_billing_event_with_token_usage(
    mock_request, billing_event_client, setup_token_usage
):
    setup_token_usage(
        input_tokens=100, output_tokens=50, model_name="claude-3-5-sonnet-20241022"
    )

    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        response_body = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
        return Response(content=json.dumps(response_body).encode(), status_code=200)

    await dummy_func(mock_request, billing_event_client=billing_event_client)

    expected_metadata = {
        "llm_operations": [
            {
                "model_id": "claude-3-5-sonnet-20241022",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        ],
        "feature_qualified_name": "ai_gateway_proxy_use",
        "feature_ai_catalog_item": False,
    }

    billing_event_client.track_billing_event.assert_called_once_with(
        mock_request.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category="ai_gateway.api.v1.proxy.request",
        unit_of_measure="request",
        quantity=1,
        metadata=expected_metadata,
    )


@pytest.mark.asyncio
async def test_track_billing_event_with_cache_tokens(
    mock_request, billing_event_client, setup_token_usage
):
    setup_token_usage(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=20,
        cache_read_input_tokens=30,
        model_name="claude-3-5-sonnet-20241022",
    )

    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        response_body = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 30,
            },
        }
        return Response(content=json.dumps(response_body).encode(), status_code=200)

    await dummy_func(mock_request, billing_event_client=billing_event_client)

    expected_metadata = {
        "llm_operations": [
            {
                "model_id": "claude-3-5-sonnet-20241022",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 30,
            }
        ],
        "feature_qualified_name": "ai_gateway_proxy_use",
        "feature_ai_catalog_item": False,
    }

    billing_event_client.track_billing_event.assert_called_once_with(
        mock_request.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category="ai_gateway.api.v1.proxy.request",
        unit_of_measure="request",
        quantity=1,
        metadata=expected_metadata,
    )


@pytest.mark.asyncio
async def test_track_billing_event_with_exception(mock_request, billing_event_client):
    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        raise ValueError

    with pytest.raises(ValueError):
        await dummy_func(mock_request, billing_event_client=billing_event_client)

    billing_event_client.track_billing_event.assert_not_called()


@pytest.mark.asyncio
async def test_track_billing_event_with_non_200_status(
    mock_request, billing_event_client
):
    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        response_body = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            },
        }
        return Response(content=json.dumps(response_body).encode(), status_code=400)

    await dummy_func(mock_request, billing_event_client=billing_event_client)

    # Should not include metadata for non-200 responses
    billing_event_client.track_billing_event.assert_called_once_with(
        mock_request.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category="ai_gateway.api.v1.proxy.request",
        unit_of_measure="request",
        quantity=1,
        metadata=None,
    )


@pytest.mark.asyncio
async def test_track_billing_event_with_openai_format(
    mock_request, billing_event_client, setup_token_usage
):

    setup_token_usage(
        input_tokens=38,
        output_tokens=1024,
        total_tokens=1062,
        model_name="gpt-5-2025-08-07",
    )

    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        response_body = {
            "model": "gpt-5-2025-08-07",
            "usage": {
                "prompt_tokens": 38,
                "completion_tokens": 1024,
                "total_tokens": 1062,
            },
        }
        return Response(content=json.dumps(response_body).encode(), status_code=200)

    await dummy_func(mock_request, billing_event_client=billing_event_client)

    expected_metadata = {
        "llm_operations": [
            {
                "model_id": "gpt-5-2025-08-07",
                "prompt_tokens": 38,
                "completion_tokens": 1024,
                "total_tokens": 1062,
            }
        ],
        "feature_qualified_name": "ai_gateway_proxy_use",
        "feature_ai_catalog_item": False,
    }

    billing_event_client.track_billing_event.assert_called_once_with(
        mock_request.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category="ai_gateway.api.v1.proxy.request",
        unit_of_measure="request",
        quantity=1,
        metadata=expected_metadata,
    )


@pytest.mark.asyncio
async def test_track_billing_event_with_openai_cached_tokens(
    mock_request, billing_event_client, setup_token_usage
):
    setup_token_usage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cache_read_input_tokens=30,
        model_name="gpt-5-2025-08-07",
    )

    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        response_body = {
            "model": "gpt-5-2025-08-07",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {
                    "cached_tokens": 30,
                },
            },
        }
        return Response(content=json.dumps(response_body).encode(), status_code=200)

    await dummy_func(mock_request, billing_event_client=billing_event_client)

    expected_metadata = {
        "llm_operations": [
            {
                "model_id": "gpt-5-2025-08-07",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cache_read_input_tokens": 30,
            }
        ],
        "feature_qualified_name": "ai_gateway_proxy_use",
        "feature_ai_catalog_item": False,
    }

    billing_event_client.track_billing_event.assert_called_once_with(
        mock_request.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category="ai_gateway.api.v1.proxy.request",
        unit_of_measure="request",
        quantity=1,
        metadata=expected_metadata,
    )


@pytest.mark.asyncio
async def test_track_billing_event_with_openai_reasoning_tokens(
    mock_request, billing_event_client, setup_token_usage
):
    setup_token_usage(
        input_tokens=38,
        output_tokens=1024,
        total_tokens=1062,
        reasoning_tokens=1024,
        model_name="gpt-5-2025-08-07",
    )

    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        # This is the exact response format from the user's example
        response_body = {
            "id": "chatcmpl-CdJZaJTA6UgqxTVQIWH3N8itNUq9Z",
            "object": "chat.completion",
            "model": "gpt-5-2025-08-07",
            "usage": {
                "prompt_tokens": 38,
                "completion_tokens": 1024,
                "total_tokens": 1062,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {
                    "reasoning_tokens": 1024,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
        }
        return Response(content=json.dumps(response_body).encode(), status_code=200)

    await dummy_func(mock_request, billing_event_client=billing_event_client)

    expected_metadata = {
        "llm_operations": [
            {
                "model_id": "gpt-5-2025-08-07",
                "prompt_tokens": 38,
                "completion_tokens": 1024,  # Includes reasoning tokens
                "total_tokens": 1062,
                "reasoning_tokens": 1024,  # Reasoning tokens are also tracked separately
            }
        ],
        "feature_qualified_name": "ai_gateway_proxy_use",
        "feature_ai_catalog_item": False,
    }

    billing_event_client.track_billing_event.assert_called_once_with(
        mock_request.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category="ai_gateway.api.v1.proxy.request",
        unit_of_measure="request",
        quantity=1,
        metadata=expected_metadata,
    )


@pytest.mark.asyncio
async def test_verify_project_namespace_metadata_saas_success(mock_request):
    """Test successful verification for SaaS with matching project/namespace IDs."""
    user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                gitlab_realm="saas",
                extra={
                    "gitlab_project_id": "123",
                    "gitlab_namespace_id": "456",
                    "gitlab_root_namespace_id": "789",
                },
            ),
        )
    )
    mock_request.user = user

    event_context = EventContext(
        project_id=123,
        namespace_id=456,
        ultimate_parent_namespace_id=789,
    )
    current_event_context.set(event_context)

    @verify_project_namespace_metadata()
    async def dummy_func(request, *args, **kwargs):  # pylint: disable=unused-argument
        return Response(content=b'{"message": "success"}', status_code=200)

    response = await dummy_func(mock_request)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_verify_project_namespace_metadata_saas_project_mismatch(mock_request):
    """Test SaaS verification fails when project ID doesn't match."""
    user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                gitlab_realm="saas",
                extra={
                    "gitlab_project_id": "999",
                    "gitlab_namespace_id": "456",
                    "gitlab_root_namespace_id": "789",
                },
            ),
        )
    )
    mock_request.user = user

    event_context = EventContext(
        project_id=123,
        namespace_id=456,
        ultimate_parent_namespace_id=789,
    )
    current_event_context.set(event_context)

    @verify_project_namespace_metadata()
    async def dummy_func(request, *args, **kwargs):  # pylint: disable=unused-argument
        return Response(content=b'{"message": "success"}', status_code=200)

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request)

    assert exc_info.value.status_code == 403
    assert "project id mismatch" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_project_namespace_metadata_saas_namespace_mismatch(mock_request):
    """Test SaaS verification fails when namespace ID doesn't match."""
    user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                gitlab_realm="saas",
                extra={
                    "gitlab_project_id": "123",
                    "gitlab_namespace_id": "999",
                    "gitlab_root_namespace_id": "789",
                },
            ),
        )
    )
    mock_request.user = user

    event_context = EventContext(
        project_id=123,
        namespace_id=456,
        ultimate_parent_namespace_id=789,
    )
    current_event_context.set(event_context)

    @verify_project_namespace_metadata()
    async def dummy_func(request, *args, **kwargs):  # pylint: disable=unused-argument
        return Response(content=b'{"message": "success"}', status_code=200)

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request)

    assert exc_info.value.status_code == 403
    assert "namespace id mismatch" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_project_namespace_metadata_saas_root_namespace_mismatch(
    mock_request,
):
    """Test SaaS verification fails when root namespace ID doesn't match."""
    user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                gitlab_realm="saas",
                extra={
                    "gitlab_project_id": "123",
                    "gitlab_namespace_id": "456",
                    "gitlab_root_namespace_id": "999",
                },
            ),
        )
    )
    mock_request.user = user

    event_context = EventContext(
        project_id=123,
        namespace_id=456,
        ultimate_parent_namespace_id=789,
    )
    current_event_context.set(event_context)

    @verify_project_namespace_metadata()
    async def dummy_func(request, *args, **kwargs):  # pylint: disable=unused-argument
        return Response(content=b'{"message": "success"}', status_code=200)

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request)

    assert exc_info.value.status_code == 403
    assert "root namespace id mismatch" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_project_namespace_metadata_self_managed_success(mock_request):
    """Test successful verification for self-managed with matching instance UID."""
    user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                gitlab_realm="self-managed",
                gitlab_instance_uid="instance-uid-123",
                extra={"gitlab_instance_uid": "instance-uid-123"},
            ),
        )
    )
    mock_request.user = user

    event_context = EventContext(instance_id="instance-uid-123")
    current_event_context.set(event_context)

    @verify_project_namespace_metadata()
    async def dummy_func(request, *args, **kwargs):  # pylint: disable=unused-argument
        return Response(content=b'{"message": "success"}', status_code=200)

    response = await dummy_func(mock_request)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_verify_project_namespace_metadata_self_managed_instance_mismatch(
    mock_request,
):
    """Test self-managed verification fails when instance UID doesn't match."""
    user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                gitlab_realm="self-managed",
                gitlab_instance_uid="wrong-instance-uid",
                extra={"gitlab_instance_uid": "wrong-instance-uid"},
            ),
        )
    )
    mock_request.user = user

    event_context = EventContext(instance_id="instance-uid-123")
    current_event_context.set(event_context)

    @verify_project_namespace_metadata()
    async def dummy_func(request, *args, **kwargs):  # pylint: disable=unused-argument
        return Response(content=b'{"message": "success"}', status_code=200)

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request)

    assert exc_info.value.status_code == 403
    assert "instance uid mismatch" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_project_namespace_metadata_self_managed_ignores_project_ids(
    mock_request,
):
    """Test self-managed verification ignores project/namespace IDs in extra claims."""
    user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                gitlab_realm="self-managed",
                gitlab_instance_uid="instance-uid-123",
                extra={
                    "gitlab_instance_uid": "instance-uid-123",
                    "gitlab_project_id": "999",  # Should be ignored
                    "gitlab_namespace_id": "999",  # Should be ignored
                    "gitlab_root_namespace_id": "999",  # Should be ignored
                },
            ),
        )
    )
    mock_request.user = user

    event_context = EventContext(
        instance_id="instance-uid-123",
        project_id=123,  # Different from extra claims
        namespace_id=456,  # Different from extra claims
        ultimate_parent_namespace_id=789,  # Different from extra claims
    )
    current_event_context.set(event_context)

    @verify_project_namespace_metadata()
    async def dummy_func(request, *args, **kwargs):  # pylint: disable=unused-argument
        return Response(content=b'{"message": "success"}', status_code=200)

    # Should succeed because self-managed only checks instance_uid
    response = await dummy_func(mock_request)
    assert response.status_code == 200
