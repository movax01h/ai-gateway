import json

import pytest

from duo_workflow_service.tools.todo_write import (
    TodoItem,
    TodoStatus,
    TodoWrite,
    TodoWriteInput,
)


@pytest.fixture(name="tool")
def tool_fixture() -> TodoWrite:
    return TodoWrite()


@pytest.mark.asyncio
async def test_todo_write_creates_tasks(tool: TodoWrite):
    todos = [
        TodoItem(description="First task", status=TodoStatus.PENDING),
        TodoItem(description="Second task", status=TodoStatus.IN_PROGRESS),
        TodoItem(description="Third task", status=TodoStatus.COMPLETED),
    ]

    result = await tool._arun(todos=todos)
    items = json.loads(result)

    assert items == [
        {"description": "First task", "status": "pending"},
        {"description": "Second task", "status": "in_progress"},
        {"description": "Third task", "status": "completed"},
    ]


@pytest.mark.asyncio
async def test_todo_write_returns_full_json(tool: TodoWrite):
    todos = [TodoItem(description="Do something", status=TodoStatus.PENDING)]

    result = await tool._arun(todos=todos)
    items = json.loads(result)

    assert items == [{"description": "Do something", "status": "pending"}]


@pytest.mark.asyncio
async def test_todo_write_empty_list(tool: TodoWrite):
    result = await tool._arun(todos=[])
    assert json.loads(result) == []


@pytest.mark.asyncio
async def test_todo_write_replaces_on_each_call(tool: TodoWrite):
    first = await tool._arun(
        todos=[TodoItem(description="Old task", status=TodoStatus.PENDING)]
    )
    second = await tool._arun(
        todos=[TodoItem(description="New task", status=TodoStatus.IN_PROGRESS)]
    )

    assert json.loads(first) == [{"description": "Old task", "status": "pending"}]
    assert json.loads(second) == [{"description": "New task", "status": "in_progress"}]


def test_todo_write_format_display_message_counts_non_completed():
    tool = TodoWrite()
    args = TodoWriteInput(
        todos=[
            TodoItem(description="Task 1", status=TodoStatus.PENDING),
            TodoItem(description="Task 2", status=TodoStatus.IN_PROGRESS),
            TodoItem(description="Task 3", status=TodoStatus.COMPLETED),
            TodoItem(description="Task 4", status=TodoStatus.CANCELLED),
        ]
    )

    assert tool.format_display_message(args) == "2 todos remaining"


def test_todo_write_format_display_message_all_completed():
    tool = TodoWrite()
    args = TodoWriteInput(
        todos=[
            TodoItem(description="Task 1", status=TodoStatus.COMPLETED),
            TodoItem(description="Task 2", status=TodoStatus.COMPLETED),
        ]
    )

    assert tool.format_display_message(args) == "0 todos remaining"
