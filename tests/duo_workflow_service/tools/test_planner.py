import pytest

from duo_workflow_service.tools.planner import (
    AddNewTask,
    AddNewTaskInput,
    CreatePlan,
    CreatePlanInput,
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
        (
            RemoveTask,
            RemoveTaskInput,
            {"task_id": "1", "description": "Task 1"},
            "Task with ID 1 removed",
        ),
        (
            UpdateTaskDescription,
            UpdateTaskDescriptionInput,
            {"task_id": "1", "new_description": "Updated task"},
            "Task with ID 1 updated with description: Updated task",
        ),
        (
            CreatePlan,
            CreatePlanInput,
            {"tasks": ["Task 1", "Task 2", "Task 3"]},
            "Plan created successfully",
        ),
    ],
)
def test_tool_run(tool_class, input_class, input_data, expected_result):
    tool = tool_class()
    if input_class:
        input_instance = input_class(**input_data)
        if isinstance(tool, AddNewTask):
            result = tool._run(description=input_instance.description)
        elif isinstance(tool, CreatePlan):
            result = tool._run(tasks=input_instance.tasks)
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
    input_data = SetTaskStatusInput(
        task_id=task_id, status=status, description="test description"
    )
    result = set_task_status._run(task_id=input_data.task_id, status=input_data.status)
    assert result == expected_result


def test_add_new_task_format_display_message():
    tool = AddNewTask(description="Add new task")

    input_data = AddNewTaskInput(description="Create new feature")

    message = tool.format_display_message(input_data)

    expected_message = "Add new task to the plan: Create new feature"
    assert message == expected_message


def test_remove_task_format_display_message():
    tool = RemoveTask(description="Remove task")

    input_data = RemoveTaskInput(task_id="task-1", description="Task 1")

    message = tool.format_display_message(input_data)

    expected_message = "Remove task 'Task 1'"
    assert message == expected_message


def test_update_task_description_format_display_message():
    tool = UpdateTaskDescription(description="Update task description")

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
    tool = SetTaskStatus(description="Set task status")

    input_data = SetTaskStatusInput(
        task_id=task_id,
        status=status,
        description=description,
    )

    message = tool.format_display_message(input_data)
    assert message == expected_result


def test_create_plan():
    create_plan = CreatePlan(description="Create a plan")
    tasks = ["Task 1", "Task 2", "Task 3"]
    result = create_plan._run(tasks=tasks)
    assert result == "Plan created successfully"


def test_create_plan_format_display_message():
    create_plan = CreatePlan(description="Create a plan")
    tasks = ["Task 1", "Task 2", "Task 3"]
    input_data = CreatePlanInput(tasks=tasks)

    message = create_plan.format_display_message(input_data)
    assert message == "Create a plan: Task 1, Task 2, Task 3..."


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
