import json
from abc import abstractmethod
from typing import Any, List, Optional, Type

from langchain_core.messages import ToolMessage
from langgraph.types import Command as LangGraphCommand
from pydantic import BaseModel, Field

from duo_workflow_service.entities.state import Plan, Task, TaskStatus
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


def format_task_number(task_id: str) -> str:
    task_num = task_id.split("-")[-1] if "-" in task_id else task_id
    try:
        return str(int(task_num) + 1)
    except (ValueError, TypeError):
        return task_id


def format_short_task_description(
    description: str,
    word_limit: Optional[int] = None,
    char_limit: int = 100,
    suffix: str = "...",
) -> str:

    words = description.strip().split()
    shortened_description = " ".join(words[:word_limit])

    if len(shortened_description) > char_limit:
        shortened_description = shortened_description[:char_limit].rsplit(" ", 1)[0]

    return (
        f"{shortened_description}{suffix}"
        if (word_limit and len(words) > word_limit)
        or len(shortened_description) < len(description.strip())
        else shortened_description
    )


class PlannerTool(DuoBaseTool):
    # The intended mechanism to get the tool_call_id is to use `InjectedToolCallId`, but at the moment it doesn't play
    # nicely with custom tool input schemas, so we set it manually.
    # See https://github.com/langchain-ai/langgraph/issues/5072
    tool_call_id: str | None = None
    _plan: Plan | None = None
    _tools_agent_name: str | None = None

    # Planner tools return internal workflow commands, not user content
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    @property
    def plan(self) -> Plan:
        if not self._plan:
            raise RuntimeError("plan is not set")
        return self._plan

    @plan.setter
    def plan(self, plan: Plan):
        self._plan = plan

    @property
    def tools_agent_name(self) -> str:
        if not self._tools_agent_name:
            raise RuntimeError("tools_agent_name is not set")
        return self._tools_agent_name

    @tools_agent_name.setter
    def tools_agent_name(self, tools_agent_name: str):
        self._tools_agent_name = tools_agent_name

    @abstractmethod
    async def _execute(self, *args, **kwargs):
        """Planner tool doesn't need async execution."""

    def _command(self, steps: List[Task], tool_message: str, reset: bool = False):
        return LangGraphCommand(
            update={
                "conversation_history": {
                    self.tools_agent_name: [
                        ToolMessage(
                            content=tool_message,
                            name=self.name,
                            tool_call_id=self.tool_call_id,
                        )
                    ],
                },
                "plan": Plan(steps=steps, reset=reset),
            }
        )


class AddNewTaskInput(BaseModel):
    description: str = Field(description="The description of the new task to add")


class AddNewTask(PlannerTool):
    name: str = "add_new_task"
    description: str = """Add a task to a plan for a workflow.
    A plan consists of a list of tasks and the status of each task.
    This tool adds a task to the list of tasks but should never update the status of a task."""

    args_schema: Type[BaseModel] = AddNewTaskInput

    async def _execute(self, description: str) -> LangGraphCommand:
        new_task = Task(
            id=f"task-{len(self.plan['steps'])}",
            description=description,
            status=TaskStatus.NOT_STARTED,
        )

        return self._command([new_task], f"Step added: {new_task['id']}")

    def format_display_message(
        self, args: AddNewTaskInput, _tool_response: Any = None
    ) -> str:
        return f"Add new task to the plan: {format_short_task_description(args.description, char_limit=100)}"


class RemoveTaskInput(BaseModel):
    task_id: str = Field(description="The ID of the task to remove")
    description: str = Field(description="The description of the task to remove")


class RemoveTask(PlannerTool):
    name: str = "remove_task"
    description: str = """Remove a task from a plan based on its ID.
    A plan consists of a list of tasks and the status of each task.
    This tool removes a task from the list of tasks."""
    args_schema: Type[BaseModel] = RemoveTaskInput

    async def _execute(
        self, task_id: str, description: str  # pylint: disable=unused-argument
    ) -> LangGraphCommand:
        steps: List[Task]

        step = next(
            (step for step in self.plan["steps"] if step["id"] == task_id), None
        )

        if step:
            step["delete"] = True
            steps = [step]
        else:
            steps = []  # The step already doesn't exist, so no update is needed

        return self._command(steps, f"Task removed: {task_id}")

    def format_display_message(
        self, args: RemoveTaskInput, _tool_response: Any = None
    ) -> str:
        short_description = format_short_task_description(
            args.description, word_limit=5, char_limit=50
        )
        return f"Remove task '{short_description}'"


class UpdateTaskDescriptionInput(BaseModel):
    task_id: str = Field(description="The ID of the task to update")
    new_description: str = Field(description="The new description for the task")


class UpdateTaskDescription(PlannerTool):
    name: str = "update_task_description"
    description: str = """Update the description of a task in the plan.
    A plan consists of a list of tasks and the status of each task.
    This tool updates the description of a task but should never update the status of a task."""
    args_schema: Type[BaseModel] = UpdateTaskDescriptionInput

    async def _execute(
        self, task_id: str, new_description: str
    ) -> LangGraphCommand | str:
        for step in self.plan["steps"]:
            if step["id"] == task_id:
                if new_description:
                    step["description"] = new_description
                    return self._command([step], f"Task updated: {task_id}")

        return f"Task not found: {task_id}"

    def format_display_message(
        self, args: UpdateTaskDescriptionInput, _tool_response: Any = None
    ) -> str:
        short_new_description = format_short_task_description(
            args.new_description, word_limit=5, char_limit=50
        )
        return f"Update description for task '{short_new_description}'"


class GetPlan(PlannerTool):
    name: str = "get_plan"
    description: str = """Fetch a list of tasks for a workflow.
    A plan consists of a list of tasks and the status of each task."""

    async def _execute(self) -> str:
        return json.dumps(self.plan["steps"])


class SetTaskStatusInput(BaseModel):
    task_id: str = Field(description="The ID of the task to update")
    status: str = Field(
        description="""The status of the task.
                        The status can be `Not Started`, `In Progress`,
                        `Completed` or `Cancelled`"""
    )
    description: str = Field(description="A description of the task for context")


class SetTaskStatus(PlannerTool):
    name: str = "set_task_status"
    description: str = "Set the status of a single task in the plan"
    args_schema: Type[BaseModel] = SetTaskStatusInput

    async def _execute(
        self,
        task_id: str,
        status: str,
        description: str,  # pylint: disable=unused-argument
    ) -> LangGraphCommand | str:
        for step in self.plan["steps"]:
            if step["id"] == task_id:
                step["status"] = TaskStatus(status)
                return self._command([step], f"Task status set: {task_id} - {status}")

        return f"Task not found: {task_id}"

    def format_display_message(
        self, args: SetTaskStatusInput, _tool_response: Any = None
    ) -> str:
        task_description = format_short_task_description(
            args.description, word_limit=5, char_limit=50
        )
        return f"Set task '{task_description}' to '{args.status}'"


class CreatePlanInput(BaseModel):
    tasks: List[str] = Field(
        description=(
            "A list of tasks, where each task is a separate string element in the array. "
            "Do NOT provide a single multi-line string. "
            "Example: ['Check repo structure', 'Run tests', 'Fix warnings']"
        ),
    )


class CreatePlan(PlannerTool):
    name: str = "create_plan"
    description: str = """Create a list of tasks for the plan.
    The tasks you provide here will set the tasks in the current plan.
    Please provide all the tasks that you want to show to the user.
    Tasks should be formatted in an array where each task is a string.
    """

    args_schema: Type[BaseModel] = CreatePlanInput

    async def _execute(self, tasks: List[str]) -> LangGraphCommand:
        steps: List[Task] = []
        for i, task_description in enumerate(tasks):
            steps.append(
                Task(
                    id=f"task-{i}",
                    description=task_description,
                    status=TaskStatus.NOT_STARTED,
                )
            )

        return self._command(steps, "Plan created", reset=True)

    def format_display_message(
        self, args: CreatePlanInput, _tool_response: Any = None
    ) -> str:
        return f"Create plan with {len(args.tasks)} tasks"
