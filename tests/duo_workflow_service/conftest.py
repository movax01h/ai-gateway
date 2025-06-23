from datetime import datetime, timezone

import pytest

from duo_workflow_service.entities.state import Plan, Task


@pytest.fixture
def plan_steps() -> list[Task]:
    return []


@pytest.fixture
def plan(plan_steps: list[Task]) -> Plan:
    return Plan(steps=plan_steps)


@pytest.fixture
def mock_now() -> datetime:
    return datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
