from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import Request
from starlette_context import context, request_cycle_context

from ai_gateway.api.middleware import FeatureFlagMiddleware


@pytest.fixture
def feature_flag_middleware(mock_app, disallowed_flags):
    return FeatureFlagMiddleware(mock_app, disallowed_flags)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers,disallowed_flags,expected_flags",
    [
        (
            [
                (b"x-gitlab-enabled-feature-flags", b"feature_a,feature_b,feature_c"),
            ],
            {},
            {"feature_a", "feature_b", "feature_c"},
        ),
        (
            [
                (b"x-gitlab-enabled-feature-flags", b"feature_a,feature_b,feature_c"),
                (b"x-gitlab-realm", b"self-managed"),
            ],
            {"self-managed": {"feature_a"}},
            {"feature_b", "feature_c"},
        ),
    ],
)
async def test_middleware_feature_flag(
    feature_flag_middleware, headers, expected_flags
):
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": headers,
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with patch(
        "ai_gateway.api.middleware.feature_flag.current_feature_flag_context"
    ) as mock_feature_flag_context, request_cycle_context({}):
        await feature_flag_middleware(scope, receive, send)

        mock_feature_flag_context.set.assert_called_once_with(expected_flags)

        assert set(context["enabled_feature_flags"].split(",")) == expected_flags

    feature_flag_middleware.app.assert_called_once_with(scope, receive, send)
