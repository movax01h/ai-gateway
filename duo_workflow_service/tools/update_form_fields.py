import json
from typing import Any, Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class UpdateFormFieldsInput(BaseModel):
    """Input schema for the UpdateFormFields tool.

    Attributes:
        form_id: Identifier of the form being updated, as provided by the flow context.
            Pinned by the system prompt — never invented by the LLM.
        select: Field/option names to select or enable in the form.
        clear: Field/option names to clear or disable in the form.
    """

    form_id: str = Field(
        min_length=1,
        description="Identifier of the form being updated, as provided by the flow context",
    )
    select: list[str] = Field(
        default_factory=list,
        description="Field/option names to select or enable in the form",
    )
    clear: list[str] = Field(
        default_factory=list,
        description="Field/option names to clear or disable in the form",
    )


class UpdateFormFields(DuoBaseTool):
    """Generic tool for updating a UI form by selecting or clearing named fields.

    Designed to be reusable across any foundational agent that edits a GitLab UI
    form. The caller must supply ``form_id`` as instructed by the system prompt;
    the prompt receives it from the flow's ``form_context`` additional_context
    envelope.
    """

    name: str = "update_form_fields"
    description: str = (
        "Update a UI form by selecting or clearing named fields or options. "
        "Use this tool to apply changes to any form in the GitLab UI. "
        "Always set form_id to the value provided in the system prompt."
    )
    args_schema: Type[BaseModel] = UpdateFormFieldsInput

    async def _execute(
        self,
        form_id: str,
        select: list[str] | None = None,
        clear: list[str] | None = None,
    ) -> str:
        return json.dumps(
            {"form_id": form_id, "select": select or [], "clear": clear or []}
        )

    def format_display_message(
        self, args: UpdateFormFieldsInput, _tool_response: Any = None
    ) -> str:
        parts = []
        if args.select:
            parts.append(f"Select: {', '.join(args.select)}")
        if args.clear:
            parts.append(f"Clear: {', '.join(args.clear)}")
        if not parts:
            return "No form field changes"
        return "Update form fields — " + "; ".join(parts)
