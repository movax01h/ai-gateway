# pylint: disable=file-naming-for-tests,unused-import

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from duo_workflow_service.agents import PlanTerminatorAgent
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    Plan,
    Task,
    TaskStatus,
    WorkflowState,
    WorkflowStatusEnum,
)


class TestPlanTerminatorAgent:
    @pytest.fixture
    def base_workflow_state(self) -> WorkflowState:
        return WorkflowState(
            status=WorkflowStatusEnum.ERROR,
            conversation_history={},
            handover=[],
            last_human_input=None,
            plan=Plan(steps=[]),
            ui_chat_log=[],
        )

    @dataclass
    class PlanTerminatorTestCase:
        initial_steps: List[Task]
        expected_updates: List[TaskStatus]
        needs_update: bool

    @pytest.mark.parametrize(
        "test_case",
        [
            PlanTerminatorTestCase(
                initial_steps=[
                    {
                        "id": "1",
                        "description": "Task 1",
                        "status": TaskStatus.COMPLETED,
                    },
                    {
                        "id": "2",
                        "description": "Task 2",
                        "status": TaskStatus.IN_PROGRESS,
                    },
                    {
                        "id": "3",
                        "description": "Task 3",
                        "status": TaskStatus.NOT_STARTED,
                    },
                ],
                expected_updates=[
                    TaskStatus.COMPLETED,
                    TaskStatus.CANCELLED,
                    TaskStatus.CANCELLED,
                ],
                needs_update=True,
            ),
            PlanTerminatorTestCase(
                initial_steps=[
                    {
                        "id": "1",
                        "description": "Task 1",
                        "status": TaskStatus.COMPLETED,
                    },
                    {
                        "id": "2",
                        "description": "Task 2",
                        "status": TaskStatus.COMPLETED,
                    },
                ],
                expected_updates=[],
                needs_update=False,
            ),
            PlanTerminatorTestCase(
                initial_steps=[
                    {
                        "id": "1",
                        "description": "Task 1",
                        "status": TaskStatus.CANCELLED,
                    },
                    {
                        "id": "2",
                        "description": "Task 2",
                        "status": TaskStatus.NOT_STARTED,
                    },
                ],
                expected_updates=[
                    TaskStatus.CANCELLED,
                    TaskStatus.CANCELLED,
                ],
                needs_update=True,
            ),
        ],
    )
    @pytest.mark.asyncio
    @patch("duo_workflow_service.agents.plan_terminator.datetime")
    async def test_run_with_tasks(
        self,
        mock_datetime,
        base_workflow_state: WorkflowState,
        test_case: PlanTerminatorTestCase,
    ):
        mock_now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now
        mock_datetime.timezone = timezone

        workflow_state = base_workflow_state.copy()
        workflow_state["plan"] = Plan(steps=test_case.initial_steps)

        plan_terminator = PlanTerminatorAgent(workflow_id="123")
        result = await plan_terminator.run(workflow_state)

        if not test_case.needs_update:
            assert result == {"plan": workflow_state["plan"]}
            assert "ui_chat_log" not in result
        else:
            for i, expected_status in enumerate(test_case.expected_updates):
                assert result["plan"]["steps"][i]["status"] == expected_status

            for original, resulting in zip(
                test_case.initial_steps, result["plan"]["steps"]
            ):
                assert resulting["description"] == original["description"]
                assert resulting["id"] == original["id"]

            assert "ui_chat_log" in result
            assert len(result["ui_chat_log"]) == 1
            chat_message = result["ui_chat_log"][0]
            assert chat_message["message_type"] == MessageTypeEnum.WORKFLOW_END
            assert chat_message["content"] == (
                "Your request was valid but Workflow failed to complete it. Please try again."
            )
            assert chat_message["timestamp"] == "2025-01-01T12:00:00+00:00"

    @pytest.mark.parametrize(
        "input_state,description",
        [
            ({"plan": Plan(steps=[])}, "empty steps list"),
            ({"plan": None}, "no plan"),
            (
                {
                    "plan": Plan(steps=[]),
                    "status": WorkflowStatusEnum.CANCELLED,
                },
                "cancelled with empty steps",
            ),
        ],
        ids=lambda param: param[1] if isinstance(param, tuple) else "default",
    )
    @pytest.mark.asyncio
    async def test_run_edge_cases(
        self, base_workflow_state: WorkflowState, input_state, description
    ):
        workflow_state = base_workflow_state.copy()
        workflow_state.update(input_state)

        plan_terminator = PlanTerminatorAgent(workflow_id="123")
        result = await plan_terminator.run(workflow_state)

        expected: Dict[str, Any] = {"plan": {"steps": []}}
        assert (
            result == expected
        ), f"Expected plan with empty steps for case: {description}"
        assert "ui_chat_log" not in result
