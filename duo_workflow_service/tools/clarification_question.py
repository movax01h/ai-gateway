import json
from typing import Any, List, Type

from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

__all__ = ["ClarificationQuestionTool"]


class ClarificationOption(BaseModel):
    """A single mutually-exclusive choice presented to the user in a clarification question."""

    id: str = Field(description="Unique identifier for this option.")
    label: str = Field(description="Short display label shown to the user.")
    description: str = Field(
        description="One or two sentences explaining the option and its trade-offs."
    )
    recommended: bool = Field(
        default=False,
        description="Set to true for the single best default choice when one exists.",
    )


class ClarificationQuestionInput(BaseModel):
    """Input schema for ``ClarificationQuestionTool``: a question with 2-3 options."""

    question: str = Field(
        description="A single concise sentence stating exactly what needs to be decided."
    )
    options: List[ClarificationOption] = Field(
        min_length=2,
        max_length=3,
        description="Between 2 and 3 distinct, mutually-exclusive choices for the user.",
    )


class ClarificationQuestionTool(DuoBaseTool):
    """Chat-agent tool for asking the user a structured multiple-choice clarification question."""

    name: str = "clarification_question"
    description: str = """Ask the user a single structured clarifying question with labelled options.

    Use this tool when the user's request is ambiguous and proceeding
    without clarification would produce the wrong outcome or require
    significant rework.

    Rules:
    - Ask exactly ONE question per tool call. Never call this tool more
    than once before receiving the user's answer.
    - Provide 2-3 distinct, mutually-exclusive options. Mark at most one
    as recommended=true when there is a clear best default.
    - After calling this tool you MUST stop immediately. Do not call any
    other tools, do not continue the task, and do not output any
    further text. The question and options will be presented to the
    user via a tailored UI; wait for their selection before taking any
    further action.

    Do NOT use this tool when:
    - The user has already provided enough context to proceed.
    - The ambiguity is minor and can be resolved with a reasonable default assumption.
    - You would be repeating a question already answered earlier in the conversation.
    """
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL
    args_schema: Type[BaseModel] = ClarificationQuestionInput

    async def _execute(
        self,
        question: str,
        options: List[ClarificationOption],
        **_kwargs: Any,
    ) -> str:
        return json.dumps(
            {
                "question": question,
                "options": [opt.model_dump() for opt in options],
            }
        )

    def format_display_message(
        self, args: ClarificationQuestionInput, _tool_response: Any = None
    ) -> str:
        return f"Asking clarification: {args.question}"
