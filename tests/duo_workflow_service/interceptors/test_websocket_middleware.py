from unittest.mock import MagicMock

import pytest
from fastapi import WebSocket

from duo_workflow_service.interceptors.websocket_middleware import (
    MiddlewareChain,
    WebSocketMiddleware,
)


# Mock middleware for testing that extends WebSocketMiddleware
class MockMiddleware(WebSocketMiddleware):
    """Mock middleware for testing."""

    def __init__(self):
        # Initialize tracking variables
        self.called = False
        self.websocket = None

    async def __call__(self, websocket: WebSocket):
        """Record that the middleware was called and with what WebSocket."""
        self.called = True
        self.websocket = websocket


@pytest.fixture
def mock_websocket():
    """Fixture that creates a mock WebSocket object."""
    return MagicMock(spec=WebSocket)


class TestMiddlewareChain:
    """Tests for the MiddlewareChain class."""

    @pytest.mark.parametrize(
        "test_case, middleware_count",
        [
            ("no_middlewares", 0),
            ("single_middleware", 1),
            ("multiple_middlewares", 3),
        ],
    )
    @pytest.mark.asyncio
    async def test_middleware_chain_execution(
        self, mock_websocket, test_case, middleware_count
    ):
        """Test that all middlewares in the chain are executed."""
        # Create a list of mock middlewares
        mock_middlewares = [MockMiddleware() for _ in range(middleware_count)]

        # Create the middleware chain
        chain = MiddlewareChain(mock_middlewares)

        # Run the middleware chain
        await chain(mock_websocket)

        # Verify each middleware was called with the WebSocket
        for mock_middleware in mock_middlewares:
            assert mock_middleware.called is True
            assert mock_middleware.websocket is mock_websocket

    @pytest.mark.asyncio
    async def test_middleware_chain_execution_order(self, mock_websocket):
        """Test that middlewares are executed in the order they're added."""
        # Create a list to track execution order
        execution_order = []

        # Create a helper middleware class that records execution order
        class OrderTrackerMiddleware(WebSocketMiddleware):
            def __init__(self, index):
                self.index = index

            async def __call__(self, websocket: WebSocket):
                execution_order.append(self.index)

        # Create middlewares with indices
        middlewares = [OrderTrackerMiddleware(i) for i in range(3)]

        # Create the middleware chain
        chain = MiddlewareChain(middlewares)

        # Run the middleware chain
        await chain(mock_websocket)

        # Verify execution order matches the order middlewares were added
        assert execution_order == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_middleware_exception_propagation(self, mock_websocket):
        """Test that exceptions from middlewares propagate and stop the chain."""

        # Create a middleware that raises an exception
        class ExceptionMiddleware(WebSocketMiddleware):
            async def __call__(self, websocket: WebSocket):
                raise ValueError("Test exception")

        # Create a list of tracked middlewares to check execution
        first_middleware = MockMiddleware()
        last_middleware = MockMiddleware()

        # Create middlewares list with the exception middleware in the middle
        middlewares = [
            first_middleware,  # Should be called
            ExceptionMiddleware(),  # Should raise exception
            last_middleware,  # Should NOT be called due to exception
        ]

        # Create the middleware chain
        chain = MiddlewareChain(middlewares)

        # Run the middleware chain and expect an exception
        with pytest.raises(ValueError, match="Test exception"):
            await chain(mock_websocket)

        # Verify only the first middleware was called
        assert first_middleware.called is True
        assert last_middleware.called is False
