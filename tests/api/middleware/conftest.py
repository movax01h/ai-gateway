from unittest.mock import AsyncMock

import pytest


@pytest.fixture(name="mock_app")
def mock_app_fixture():
    return AsyncMock()


@pytest.fixture(name="disallowed_flags")
def disallowed_flags_fixture():
    return {}
