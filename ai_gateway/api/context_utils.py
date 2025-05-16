from enum import StrEnum
from typing import Optional

from starlette_context import context

from ai_gateway.model_metadata import TypeModelMetadata


class GitLabAiRequestType(StrEnum):
    COMPLETIONS = "completions"
    CHAT = "chat"
    GENERATIONS = "generations"
    SUGGESTIONS = "suggestions"


def populate_ai_metadata_in_context(
    model_metadata: Optional[TypeModelMetadata],
    request_type: str,
    feature_id: Optional[str] = None,
):

    if model_metadata:
        context["model_provider"] = model_metadata.provider
        context["model_name"] = model_metadata.name
        context["model_identifier"] = (
            model_metadata.identifier
            if hasattr(model_metadata, "identifier")
            else model_metadata.name
        )
    else:
        context["model_provider"] = "unknown"
        context["model_name"] = "unknown"
        context["model_identifier"] = "unknown"

    context["feature_id"] = feature_id or context.get("meta.unit_primitive", "unknown")
    context["request_type"] = request_type
