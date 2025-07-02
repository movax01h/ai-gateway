import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from duo_workflow_service.components.tools_registry import ToolMetadata
from duo_workflow_service.entities.state import Plan, Task
from duo_workflow_service.gitlab.http_client import GitlabHttpClient


@pytest.fixture
def plan_steps() -> list[Task]:
    return []


@pytest.fixture
def plan(plan_steps: list[Task]) -> Plan:
    return Plan(steps=plan_steps)


@pytest.fixture
def mock_now() -> datetime:
    return datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="function")
def gl_http_client():
    return AsyncMock(spec=GitlabHttpClient)


@pytest.fixture(scope="function")
def tool_metadata(gl_http_client):
    return ToolMetadata(
        outbox=MagicMock(spec=asyncio.Queue),
        inbox=MagicMock(spec=asyncio.Queue),
        gitlab_client=gl_http_client,
        gitlab_host="gitlab.example.com",
    )
