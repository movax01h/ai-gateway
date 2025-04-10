from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class Task(BaseModel):
    id: str = Field(description="A unique identifier for the task")
    description: str = Field(description="A description of what the task is")
    status: str = Field(
        description="""The status of the task.
                        The status can be `Not Started`, `In Progress`,
                        `Completed` or `Cancelled`"""
    )


def format_task_number(task_id: str) -> str:
    task_num = task_id.split("-")[-1] if "-" in task_id else task_id
    try:
        return str(int(task_num) + 1)
    except (ValueError, TypeError):
        return task_id


class AddNewTaskInput(BaseModel):
    description: str = Field(description="The description of the new task to add")


class AddNewTask(BaseTool):
    name: str = "add_new_task"
    description: str = """Add a task to a plan for a workflow.
    A plan consists of a list of tasks and the status of each task.
    This tool adds a task to the list of tasks but should never update the status of a task."""

    args_schema: Type[BaseModel] = AddNewTaskInput

    def _run(self, description: str) -> str:
        return "New task added"

    def format_display_message(self, args: AddNewTaskInput) -> str:
        return f"Add new task to the plan: {args.description[:100]}..."


class RemoveTaskInput(BaseModel):
    task_id: str = Field(description="The ID of the task to remove")


class RemoveTask(BaseTool):
    name: str = "remove_task"
    description: str = """Remove a task from a plan based on its ID.
    A plan consists of a list of tasks and the status of each task.
    This tool removes a task from the list of tasks."""
    args_schema: Type[BaseModel] = RemoveTaskInput

    def _run(self, task_id: str) -> str:
        return f"Task with ID {task_id} removed"

    def format_display_message(self, args: RemoveTaskInput) -> str:
        task_num = format_task_number(args.task_id)
        return f"Remove task {task_num}"


class UpdateTaskDescriptionInput(BaseModel):
    task_id: str = Field(description="The ID of the task to update")
    new_description: str = Field(description="The new description for the task")


class UpdateTaskDescription(BaseTool):
    name: str = "update_task_description"
    description: str = """Update the description of a task in the plan.
    A plan consists of a list of tasks and the status of each task.
    This tool updates the description of a task but should never update the status of a task."""
    args_schema: Type[BaseModel] = UpdateTaskDescriptionInput

    def _run(self, task_id: str, new_description: str) -> str:
        return f"Task with ID {task_id} updated with description: {new_description}"

    def format_display_message(self, args: UpdateTaskDescriptionInput) -> str:
        task_num = format_task_number(args.task_id)
        return f"Update description for task {task_num}"


class GetPlan(BaseTool):
    name: str = "get_plan"
    description: str = """Fetch a list of tasks for a workflow.
    A plan consists of a list of tasks and the status of each task."""

    def _run(self) -> str:
        return "Done"


class SetTaskStatusInput(BaseModel):
    task_id: str = Field(description="The ID of the task to update")
    status: str = Field(
        description="""The status of the task.
                        The status can be `Not Started`, `In Progress`,
                        `Completed` or `Cancelled`"""
    )


class SetTaskStatus(BaseTool):
    name: str = "set_task_status"
    description: str = "Set the status of a single task in the plan"
    args_schema: Type[BaseModel] = SetTaskStatusInput

    def _run(self, task_id: str, status: str) -> str:
        return f"Status of task with ID {task_id} set to {status}"

    def format_display_message(self, args: SetTaskStatusInput) -> str:
        task_num = format_task_number(args.task_id)
        return f"Set task {task_num} to '{args.status}'"
