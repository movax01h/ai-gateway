import json
from typing import Any, cast

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from duo_workflow_service.entities.state import Plan, Task, TaskStatus
from duo_workflow_service.tools.planner import (
    AddNewTask,
    AddNewTaskInput,
    CreatePlan,
    CreatePlanInput,
    GetPlan,
    PlannerTool,
    RemoveTask,
    RemoveTaskInput,
    SetTaskStatus,
    SetTaskStatusInput,
    UpdateTaskDescription,
    UpdateTaskDescriptionInput,
    format_task_number,
)


class Tool(PlannerTool):
    name: str = "test_tool"
    description: str = "A tool for testing"

    def _run(*_args, **_kwargs):
        return "test"


@pytest.fixture
def tool(tool_class: type[PlannerTool], plan: Plan) -> PlannerTool:
    tool = tool_class()  # type: ignore[call-arg]
    tool.plan = plan
    tool.tools_agent_name = "test"
    tool.tool_call_id = "1"

    return tool


@pytest.fixture
def plan_steps() -> list[Task]:
    return [
        {"id": "task-0", "description": "Task 1", "status": TaskStatus.NOT_STARTED},
        {"id": "task-1", "description": "Task 2", "status": TaskStatus.IN_PROGRESS},
    ]


def test_plan(plan: Plan):
    tool = Tool()
    with pytest.raises(RuntimeError):
        tool.plan

    tool.plan = plan
    assert tool.plan == plan


def test_tools_agent_name():
    tool = Tool()
    with pytest.raises(RuntimeError):
        tool.tools_agent_name

    tool.tools_agent_name = "my_agent"
    assert tool.tools_agent_name == "my_agent"


@pytest.mark.parametrize("tool_class", [GetPlan])
def test_get_plan(tool: GetPlan, plan_steps: list[Task]):
    assert tool._run() == json.dumps(plan_steps)


def assert_update(
    tool: PlannerTool,
    result: Any,
    expected_message: str,
    expected_steps: list[Task],
    reset: bool = False,
):
    result = cast(Command, result).update

    assert result["conversation_history"]["test"] == [
        ToolMessage(name=tool.name, tool_call_id="1", content=expected_message)
    ]
    assert result["plan"]["steps"] == expected_steps
    assert result["plan"].get("reset", False) == reset


@pytest.mark.parametrize("tool_class", [SetTaskStatus])
@pytest.mark.parametrize(
    "task_id, status, expected_steps",
    [
        (
            "task-0",
            "In Progress",
            [
                {"id": "task-0", "description": "Task 1", "status": "In Progress"},
            ],
        ),
        (
            "task-1",
            "Completed",
            [
                {"id": "task-1", "description": "Task 2", "status": "Completed"},
            ],
        ),
    ],
)
def test_set_task_status(
    tool: SetTaskStatus, task_id: str, status: str, expected_steps: list[Task]
):
    result = tool._run(task_id=task_id, status=status, description="")

    assert_update(
        tool=tool,
        result=result,
        expected_message=f"Task status set: {task_id} - {status}",
        expected_steps=expected_steps,
    )


@pytest.mark.parametrize("tool_class", [SetTaskStatus])
def test_set_task_status_missing_task(tool: SetTaskStatus):
    result = tool._run(task_id="task-2", status="In Progress", description="")
    assert result == "Task not found: task-2"


@pytest.mark.parametrize("tool_class", [AddNewTask])
def test_add_new_task(tool: AddNewTask):
    description = "Create new feature"

    result = tool._run(description=description)

    assert_update(
        tool=tool,
        result=result,
        expected_message="Step added: task-2",
        expected_steps=[
            {
                "id": "task-2",
                "description": "Create new feature",
                "status": TaskStatus.NOT_STARTED,
            },
        ],
    )


def test_add_new_task_format_display_message():
    tool = AddNewTask()

    input_data = AddNewTaskInput(description="Create new feature")

    message = tool.format_display_message(input_data)

    expected_message = "Add new task to the plan: Create new feature"
    assert message == expected_message


@pytest.mark.parametrize("tool_class", [RemoveTask])
def test_remove_task(tool: RemoveTask):
    result = tool._run(task_id="task-0", description="Task 1")

    assert_update(
        tool=tool,
        result=result,
        expected_message="Task removed: task-0",
        expected_steps=[
            {
                "id": "task-0",
                "description": "Task 1",
                "status": TaskStatus.NOT_STARTED,
                "delete": True,
            },
        ],
    )


def test_remove_task_format_display_message():
    tool = RemoveTask()

    input_data = RemoveTaskInput(task_id="task-1", description="Task 1")

    message = tool.format_display_message(input_data)

    expected_message = "Remove task 'Task 1'"
    assert message == expected_message


@pytest.mark.parametrize("tool_class", [UpdateTaskDescription])
def test_update_task_description(tool: UpdateTaskDescription):
    task_id = "task-1"
    new_description = "Update project documentation"

    result = tool._run(task_id=task_id, new_description=new_description)

    assert_update(
        tool=tool,
        result=result,
        expected_message=f"Task updated: {task_id}",
        expected_steps=[
            {
                "id": "task-1",
                "description": new_description,
                "status": TaskStatus.IN_PROGRESS,
            },
        ],
    )


def test_update_task_description_format_display_message():
    tool = UpdateTaskDescription()

    input_data = UpdateTaskDescriptionInput(
        task_id="task-1", new_description="Update project documentation"
    )

    message = tool.format_display_message(input_data)

    expected_message = "Update description for task 'Update project documentation'"
    assert message == expected_message


@pytest.mark.parametrize(
    "task_id, status, description, expected_result",
    [
        (
            "task-1",
            "In Progress",
            "This is a test task",
            "Set task 'This is a test task' to 'In Progress'",
        ),
        (
            "task-2",
            "In Progress",
            "Thisisatestwithalongcharacterinputtomakesureitsshortened",
            "Set task 'Thisisatestwithalongcharacterinputtomakesureitssho...' to 'In Progress'",
        ),
        (
            "task-3",
            "Not Started",
            "Supercalifragilisticexpialidocious to test a long first word",
            "Set task 'Supercalifragilisticexpialidocious to test a long...' to 'Not Started'",
        ),
        (
            "task-4",
            "Completed",
            "This is a very long task description that exceeds both the word and character limits significantly",
            "Set task 'This is a very long...' to 'Completed'",
        ),
        (
            "task-5",
            "Cancelled",
            "Supercalifragilisticexpialidocious antidisestablishmentarianism",
            "Set task 'Supercalifragilisticexpialidocious...' to 'Cancelled'",
        ),
    ],
)
def test_set_task_status_format_display_message(
    task_id, status, description, expected_result
):
    tool = SetTaskStatus()

    input_data = SetTaskStatusInput(
        task_id=task_id,
        status=status,
        description=description,
    )

    message = tool.format_display_message(input_data)
    assert message == expected_result


@pytest.mark.parametrize("tool_class", [CreatePlan])
def test_create_plan(tool: CreatePlan):
    tasks = ["Task 1", "Task 2", "Task 3"]
    result = tool._run(tasks=tasks)

    assert_update(
        tool=tool,
        result=result,
        expected_message="Plan created",
        expected_steps=[
            Task(id="task-0", description="Task 1", status=TaskStatus.NOT_STARTED),
            Task(id="task-1", description="Task 2", status=TaskStatus.NOT_STARTED),
            Task(id="task-2", description="Task 3", status=TaskStatus.NOT_STARTED),
        ],
        reset=True,
    )


def test_create_plan_format_display_message():
    create_plan = CreatePlan()
    tasks = ["Task 1", "Task 2", "Task 3"]
    input_data = CreatePlanInput(tasks=tasks)

    message = create_plan.format_display_message(input_data)
    assert message == "Create plan with 3 tasks"


@pytest.mark.parametrize(
    "input_id, expected_output",
    [
        ("0", "1"),
        ("1", "2"),
        ("5", "6"),
        ("10", "11"),
        ("task-0", "1"),
        ("task-1", "2"),
        ("task-5", "6"),
        ("task-10", "11"),
        ("abc", "abc"),  # Non-numeric strings should remain unchanged
        ("task-abc", "task-abc"),  # Task ID with non-numeric part
        ("00", "1"),
        ("task-00", "1"),
    ],
)
def testformat_task_number(input_id, expected_output):
    assert format_task_number(input_id) == expected_output
