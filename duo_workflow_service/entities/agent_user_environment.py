import json
from typing import Dict, Type

from pydantic import BaseModel, ValidationError

from duo_workflow_service.workflows.type_definitions import (
    AdditionalContext,
    OsInformationContext,
    ShellInformationContext,
)

_CONTEXT_REGISTRY: Dict[Type[BaseModel], str] = {
    OsInformationContext: "os_information_context",
    ShellInformationContext: "shell_information_context",
}


def process_agent_user_environment(
    additional_contexts: list[AdditionalContext] | None = None,
) -> Dict[str, BaseModel]:
    """Process and assign contexts to appropriate fields."""

    if additional_contexts is None or len(additional_contexts) == 0:
        return {}

    contexts = {}

    for context in additional_contexts:
        if context.category != "agent_user_environment" or not context.content:
            continue

        try:
            data = json.loads(context.content)
        except json.JSONDecodeError:
            continue

        if not isinstance(data, dict):
            continue

        data_fields = set(data.keys())

        for context_type, field_name in _CONTEXT_REGISTRY.items():
            try:
                expected_fields = set(context_type.model_fields.keys())

                if data_fields != expected_fields:
                    required_fields = {
                        name
                        for name, field in context_type.model_fields.items()
                        if field.is_required()
                    }

                    if not required_fields.issubset(
                        data_fields
                    ) or not data_fields.issubset(expected_fields):
                        continue

                instance = context_type.model_validate(data)
                contexts[field_name] = instance
                break
            except ValidationError:
                continue

    return contexts
