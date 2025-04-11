import pytest

from duo_workflow_service.tools.planner import (
    AddNewTask,
    AddNewTaskInput,
    GetPlan,
    RemoveTask,
    RemoveTaskInput,
    SetTaskStatus,
    SetTaskStatusInput,
    UpdateTaskDescription,
    UpdateTaskDescriptionInput,
    format_task_number,
)


@pytest.mark.parametrize(
    "tool_class, input_class, input_data, expected_result",
    [
        (AddNewTask, AddNewTaskInput, {"description": "New task"}, "New task added"),
        (RemoveTask, RemoveTaskInput, {"task_id": "1"}, "Task with ID 1 removed"),
        (
            UpdateTaskDescription,
            UpdateTaskDescriptionInput,
            {"task_id": "1", "new_description": "Updated task"},
            "Task with ID 1 updated with description: Updated task",
        ),
    ],
)
def test_tool_run(tool_class, input_class, input_data, expected_result):
    tool = tool_class()
    if input_class:
        input_instance = input_class(**input_data)
        if isinstance(tool, AddNewTask):
            result = tool._run(description=input_instance.description)
        else:
            result = tool._run(**input_instance.model_dump())
    else:
        result = tool._run()
    assert result == expected_result


def test_get_plan():
    get_plan = GetPlan(description="test description")
    result = get_plan._run()
    assert result == "Done"


@pytest.mark.parametrize(
    "task_id, status, expected_result",
    [
        ("1", "Not Started", "Status of task with ID 1 set to Not Started"),
        ("2", "In Progress", "Status of task with ID 2 set to In Progress"),
        ("3", "Completed", "Status of task with ID 3 set to Completed"),
        ("4", "Cancelled", "Status of task with ID 4 set to Cancelled"),
    ],
)
def test_set_task_status(task_id, status, expected_result):
    set_task_status = SetTaskStatus(description="test description")
    input_data = SetTaskStatusInput(task_id=task_id, status=status)
    result = set_task_status._run(task_id=input_data.task_id, status=input_data.status)
    assert result == expected_result


def test_add_new_task_format_display_message():
    tool = AddNewTask(description="Add new task")

    input_data = AddNewTaskInput(description="Create new feature")

    message = tool.format_display_message(input_data)

    expected_message = "Add new task to the plan: Create new feature..."
    assert message == expected_message


def test_remove_task_format_display_message():
    tool = RemoveTask(description="Remove task")

    input_data = RemoveTaskInput(task_id="task-1")

    message = tool.format_display_message(input_data)

    expected_message = "Remove task 2"
    assert message == expected_message


def test_update_task_description_format_display_message():
    tool = UpdateTaskDescription(description="Update task description")

    input_data = UpdateTaskDescriptionInput(
        task_id="task-1", new_description="Update project documentation"
    )

    message = tool.format_display_message(input_data)

    expected_message = "Update description for task 2"
    assert message == expected_message


def test_set_task_status_format_display_message():
    tool = SetTaskStatus(description="Set task status")

    input_data = SetTaskStatusInput(task_id="task-1", status="In Progress")

    message = tool.format_display_message(input_data)

    expected_message = "Set task 2 to 'In Progress'"
    assert message == expected_message


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
