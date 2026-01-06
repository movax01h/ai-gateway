from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

from ai_gateway.api.middleware.route.usage_quota import has_sufficient_usage_quota
from lib.events import FeatureQualifiedNameStatic
from lib.usage_quota import InsufficientCredits, UsageQuotaEvent


@pytest.fixture
def mock_request():
    """Create a mock request object with usage_quota_service."""
    request = MagicMock(spec=Request)
    request.app = MagicMock()
    request.app.state = MagicMock()
    request.app.state.usage_quota_service = AsyncMock()
    return request


class TestDecoratorBasics:
    """Tests for basic decorator functionality."""

    @pytest.mark.asyncio
    async def test_decorator_allows_request_on_sufficient_quota(self, mock_request):
        """Test that decorator allows request when quota check passes."""

        async def test_handler(request, *args, **kwargs):
            return JSONResponse({"status": "ok"})

        decorated = has_sufficient_usage_quota(
            feature_qualified_name=FeatureQualifiedNameStatic.CODE_SUGGESTIONS,
            event=UsageQuotaEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
        )(test_handler)

        mock_request.app.state.usage_quota_service.execute = AsyncMock()

        response = await decorated(mock_request)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_decorator_returns_402_on_insufficient_credits(self, mock_request):
        """Test that decorator returns 402 when quota is exhausted."""

        async def test_handler(request, *args, **kwargs):
            return JSONResponse({"status": "ok"})

        decorated = has_sufficient_usage_quota(
            feature_qualified_name=FeatureQualifiedNameStatic.CODE_SUGGESTIONS,
            event=UsageQuotaEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
        )(test_handler)

        mock_request.app.state.usage_quota_service.execute = AsyncMock(
            side_effect=InsufficientCredits("Insufficient credits")
        )

        response = await decorated(mock_request)

        assert response.status_code == 402

    @pytest.mark.asyncio
    async def test_decorator_returns_402_response_format(self, mock_request):
        """Test that 402 response has correct JSON format."""

        async def test_handler(request, *args, **kwargs):
            return JSONResponse({"status": "ok"})

        decorated = has_sufficient_usage_quota(
            feature_qualified_name=FeatureQualifiedNameStatic.CODE_SUGGESTIONS,
            event=UsageQuotaEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
        )(test_handler)

        mock_request.app.state.usage_quota_service.execute = AsyncMock(
            side_effect=InsufficientCredits("Insufficient credits")
        )

        response = await decorated(mock_request)

        assert response.status_code == 402
        content = response.body.decode()
        assert "insufficient_credits" in content
        assert "USAGE_QUOTA_EXCEEDED" in content


class TestEventTypeResolution:
    """Tests for event type resolution."""

    @pytest.mark.asyncio
    async def test_uses_static_event_type(self, mock_request):
        """Test that decorator uses static EventType when provided."""

        async def test_handler(request, *args, **kwargs):
            return JSONResponse({"status": "ok"})

        decorated = has_sufficient_usage_quota(
            feature_qualified_name=FeatureQualifiedNameStatic.CODE_SUGGESTIONS,
            event=UsageQuotaEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
        )(test_handler)

        mock_request.app.state.usage_quota_service.execute = AsyncMock()

        response = await decorated(mock_request)

        # Verify execute was called with correct event_type
        call_args = mock_request.app.state.usage_quota_service.execute.call_args
        assert call_args[0][1] == UsageQuotaEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS

    @pytest.mark.asyncio
    async def test_uses_dynamic_event_type_resolver(self, mock_request):
        """Test that decorator uses callable resolver when provided."""

        async def resolve_event_type(payload):
            return UsageQuotaEvent.CODE_SUGGESTIONS_CODE_GENERATIONS

        async def test_handler(request, *args, payload=None, **kwargs):
            return JSONResponse({"status": "ok"})

        decorated = has_sufficient_usage_quota(
            feature_qualified_name=FeatureQualifiedNameStatic.CODE_SUGGESTIONS,
            event=resolve_event_type,
        )(test_handler)

        # Create a mock payload with model_dump method
        mock_payload = MagicMock()
        mock_payload.model_dump = MagicMock()

        mock_request.app.state.usage_quota_service.execute = AsyncMock()

        response = await decorated(mock_request, payload=mock_payload)

        # Verify execute was called with resolved event_type
        call_args = mock_request.app.state.usage_quota_service.execute.call_args
        assert call_args[0][1] == UsageQuotaEvent.CODE_SUGGESTIONS_CODE_GENERATIONS

    @pytest.mark.asyncio
    async def test_resolver_falls_back_to_none_on_error(self, mock_request):
        """Test that resolver errors are caught and None is returned."""

        async def failing_resolver(payload):
            raise ValueError("Resolver error")

        async def test_handler(request, *args, payload=None, **kwargs):
            return JSONResponse({"status": "ok"})

        decorated = has_sufficient_usage_quota(
            feature_qualified_name=FeatureQualifiedNameStatic.CODE_SUGGESTIONS,
            event=failing_resolver,
        )(test_handler)

        mock_payload = MagicMock()
        mock_payload.model_dump = MagicMock()

        mock_request.app.state.usage_quota_service.execute = AsyncMock()

        response = await decorated(mock_request, payload=mock_payload)

        # Should still execute but with None as event (which will fail)
        assert mock_request.app.state.usage_quota_service.execute.called


class TestFeatureQualifiedName:
    """Tests for feature qualified name handling."""

    @pytest.mark.asyncio
    async def test_passes_feature_qualified_name_to_service(self, mock_request):
        """Test that feature_qualified_name is passed to service."""

        async def test_handler(request, *args, **kwargs):
            return JSONResponse({"status": "ok"})

        feature_name = FeatureQualifiedNameStatic.CODE_SUGGESTIONS

        decorated = has_sufficient_usage_quota(
            feature_qualified_name=feature_name,
            event=UsageQuotaEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
        )(test_handler)

        mock_request.app.state.usage_quota_service.execute = AsyncMock()

        response = await decorated(mock_request)

        # Verify execute was called with correct feature name
        call_args = mock_request.app.state.usage_quota_service.execute.call_args
        assert call_args[0][0].feature_qualified_name == feature_name.value


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_allows_request_on_generic_exception(self, mock_request):
        """Test that decorator allows request when unexpected error occurs (fail-open)."""

        async def test_handler(request, *args, **kwargs):
            return JSONResponse({"status": "ok"})

        decorated = has_sufficient_usage_quota(
            feature_qualified_name=FeatureQualifiedNameStatic.CODE_SUGGESTIONS,
            event=UsageQuotaEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
        )(test_handler)

        mock_request.app.state.usage_quota_service.execute = AsyncMock(
            side_effect=Exception("Unexpected error")
        )

        # Should raise the exception (not fail-open for generic exceptions)
        with pytest.raises(Exception):
            await decorated(mock_request)

    @pytest.mark.asyncio
    async def test_handler_receives_all_arguments(self, mock_request):
        """Test that decorated handler receives all arguments correctly."""

        async def test_handler(request, arg1, arg2, kwarg1=None, **kwargs):
            return JSONResponse(
                {
                    "arg1": arg1,
                    "arg2": arg2,
                    "kwarg1": kwarg1,
                }
            )

        decorated = has_sufficient_usage_quota(
            feature_qualified_name=FeatureQualifiedNameStatic.CODE_SUGGESTIONS,
            event=UsageQuotaEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
        )(test_handler)

        mock_request.app.state.usage_quota_service.execute = AsyncMock()

        response = await decorated(mock_request, "value1", "value2", kwarg1="kwvalue")

        assert response.status_code == 200


class TestEventTypeEnum:
    """Tests for different EventType enum values."""

    @pytest.mark.asyncio
    async def test_supports_code_completions_event(self, mock_request):
        """Test decorator with CODE_COMPLETIONS event type."""

        async def test_handler(request, *args, **kwargs):
            return JSONResponse({"status": "ok"})

        decorated = has_sufficient_usage_quota(
            feature_qualified_name=FeatureQualifiedNameStatic.CODE_SUGGESTIONS,
            event=UsageQuotaEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
        )(test_handler)

        mock_request.app.state.usage_quota_service.execute = AsyncMock()

        response = await decorated(mock_request)

        assert response.status_code == 200
        call_args = mock_request.app.state.usage_quota_service.execute.call_args
        assert call_args[0][1] == UsageQuotaEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS

    @pytest.mark.asyncio
    async def test_supports_code_generations_event(self, mock_request):
        """Test decorator with CODE_GENERATIONS event type."""

        async def test_handler(request, *args, **kwargs):
            return JSONResponse({"status": "ok"})

        decorated = has_sufficient_usage_quota(
            feature_qualified_name=FeatureQualifiedNameStatic.CODE_SUGGESTIONS,
            event=UsageQuotaEvent.CODE_SUGGESTIONS_CODE_GENERATIONS,
        )(test_handler)

        mock_request.app.state.usage_quota_service.execute = AsyncMock()

        response = await decorated(mock_request)

        assert response.status_code == 200
        call_args = mock_request.app.state.usage_quota_service.execute.call_args
        assert call_args[0][1] == UsageQuotaEvent.CODE_SUGGESTIONS_CODE_GENERATIONS

    @pytest.mark.asyncio
    async def test_supports_duo_chat_event(self, mock_request):
        """Test decorator with DUO_CHAT event type."""

        async def test_handler(request, *args, **kwargs):
            return JSONResponse({"status": "ok"})

        decorated = has_sufficient_usage_quota(
            feature_qualified_name=FeatureQualifiedNameStatic.DUO_CHAT_CLASSIC,
            event=UsageQuotaEvent.DUO_CHAT_CLASSIC,
        )(test_handler)

        mock_request.app.state.usage_quota_service.execute = AsyncMock()

        response = await decorated(mock_request)

        assert response.status_code == 200
        call_args = mock_request.app.state.usage_quota_service.execute.call_args
        assert call_args[0][1] == UsageQuotaEvent.DUO_CHAT_CLASSIC
