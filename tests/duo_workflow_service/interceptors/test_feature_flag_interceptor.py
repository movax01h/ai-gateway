from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from duo_workflow_service.interceptors.feature_flag_interceptor import (
    X_GITLAB_ENABLED_FEATURE_FLAGS,
    X_GITLAB_REALM_HEADER,
    FeatureFlagInterceptor,
    FeatureFlagMiddleware,
    current_feature_flag_context,
)


@pytest.fixture
def reset_context():
    """Reset the context variable after each test."""
    token = current_feature_flag_context.set(set())
    yield
    current_feature_flag_context.reset(token)


@pytest.fixture
def mock_handler_call_details():
    """Create a mock for the handler_call_details."""
    details = MagicMock()
    details.invocation_metadata = ()
    return details


@pytest.fixture
def mock_continuation():
    """Create a mock for the continuation function."""
    return AsyncMock()


@pytest.fixture
def interceptor():
    return FeatureFlagInterceptor()


class TestFeatureFlagInterceptor:
    @pytest.mark.asyncio
    async def test_intercept_service_no_feature_flags(
        self, reset_context, mock_handler_call_details, mock_continuation
    ):
        """Test interceptor with no feature flags in metadata."""
        # Setup
        interceptor = FeatureFlagInterceptor()

        # Execute
        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        # Assert
        mock_continuation.assert_called_once_with(mock_handler_call_details)
        assert current_feature_flag_context.get() == {""}

    @pytest.mark.asyncio
    async def test_intercept_service_with_feature_flags(
        self, reset_context, mock_handler_call_details, mock_continuation
    ):
        """Test interceptor with feature flags in metadata."""
        # Setup
        interceptor = FeatureFlagInterceptor()
        mock_handler_call_details.invocation_metadata = [
            ("x-gitlab-enabled-feature-flags", "flag1,flag2,flag3"),
        ]

        # Execute
        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        # Assert
        mock_continuation.assert_called_once_with(mock_handler_call_details)
        assert current_feature_flag_context.get() == {"flag1", "flag2", "flag3"}

    @pytest.mark.asyncio
    async def test_intercept_service_with_disallowed_flags(
        self, reset_context, mock_handler_call_details, mock_continuation
    ):
        """Test interceptor with disallowed feature flags."""
        # Setup
        disallowed_flags = {"realm1": {"flag1", "flag3"}}
        interceptor = FeatureFlagInterceptor(disallowed_flags=disallowed_flags)
        mock_handler_call_details.invocation_metadata = [
            ("x-gitlab-enabled-feature-flags", "flag1,flag2,flag3"),
            (X_GITLAB_REALM_HEADER, "realm1"),
        ]

        # Execute
        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        # Assert
        mock_continuation.assert_called_once_with(mock_handler_call_details)
        assert current_feature_flag_context.get() == {"flag2"}

    @pytest.mark.asyncio
    async def test_intercept_service_with_unknown_realm(
        self, reset_context, mock_handler_call_details, mock_continuation
    ):
        """Test interceptor with unknown realm."""
        # Setup
        disallowed_flags = {"realm1": {"flag1", "flag3"}}
        interceptor = FeatureFlagInterceptor(disallowed_flags=disallowed_flags)
        mock_handler_call_details.invocation_metadata = [
            ("x-gitlab-enabled-feature-flags", "flag1,flag2,flag3"),
            (X_GITLAB_REALM_HEADER, "unknown_realm"),
        ]

        # Execute
        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        # Assert
        mock_continuation.assert_called_once_with(mock_handler_call_details)
        assert current_feature_flag_context.get() == {"flag1", "flag2", "flag3"}

    @pytest.mark.asyncio
    async def test_intercept_service_with_empty_feature_flags(
        self, reset_context, mock_handler_call_details, mock_continuation
    ):
        """Test interceptor with empty feature flags string."""
        # Setup
        interceptor = FeatureFlagInterceptor()
        mock_handler_call_details.invocation_metadata = [
            ("x-gitlab-enabled-feature-flags", ""),
        ]

        # Execute
        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        # Assert
        mock_continuation.assert_called_once_with(mock_handler_call_details)
        assert current_feature_flag_context.get() == {""}

    @pytest.mark.asyncio
    async def test_intercept_service_metadata_conversion(
        self, reset_context, mock_handler_call_details, mock_continuation
    ):
        """Test that metadata tuples are correctly converted to a dict."""
        # Setup
        interceptor = FeatureFlagInterceptor()
        mock_handler_call_details.invocation_metadata = [
            ("x-gitlab-enabled-feature-flags", "flag1,flag2"),
            ("x-gitlab-global-user-id", "user123"),
            ("other-header", "value"),
        ]

        # Execute
        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        # Assert
        mock_continuation.assert_called_once_with(mock_handler_call_details)
        assert current_feature_flag_context.get() == {"flag1", "flag2"}


@pytest.fixture
def mock_websocket():
    websocket = AsyncMock()
    websocket.headers = {}
    return websocket


@pytest.mark.parametrize(
    "test_case, headers, disallowed_flags, expected_flags",
    [
        ("no_feature_flags", {}, None, {""}),
        (
            "with_feature_flags",
            {X_GITLAB_ENABLED_FEATURE_FLAGS: "flag1,flag2,flag3"},
            None,
            {"flag1", "flag2", "flag3"},
        ),
        (
            "with_disallowed_flags",
            {
                X_GITLAB_ENABLED_FEATURE_FLAGS: "flag1,flag2,flag3",
                X_GITLAB_REALM_HEADER: "realm1",
            },
            {"realm1": {"flag1", "flag3"}},
            {"flag2"},
        ),
        (
            "with_unknown_realm",
            {
                X_GITLAB_ENABLED_FEATURE_FLAGS: "flag1,flag2,flag3",
                X_GITLAB_REALM_HEADER: "unknown_realm",
            },
            {"realm1": {"flag1", "flag3"}},
            {"flag1", "flag2", "flag3"},
        ),
        ("with_empty_feature_flags", {X_GITLAB_ENABLED_FEATURE_FLAGS: ""}, None, {""}),
    ],
)
@pytest.mark.asyncio
async def test_feature_flag_middleware(
    reset_context,
    mock_websocket,
    test_case,
    headers,
    disallowed_flags,
    expected_flags,
):
    """Test the FeatureFlagMiddleware with various scenarios."""
    # Setup
    middleware = FeatureFlagMiddleware(disallowed_flags=disallowed_flags)
    mock_websocket.headers = headers

    # Execute
    await middleware(mock_websocket)

    # Assert
    assert current_feature_flag_context.get() == expected_flags
