from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def mock_app():
    return AsyncMock()


@pytest.fixture
def disallowed_flags():
    return {}
