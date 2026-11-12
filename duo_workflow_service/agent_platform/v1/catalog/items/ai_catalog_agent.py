"""The AI Catalog agent kind: an agent a customer published to the AI Catalog.

GitLab resolves the agent by id and version and serves the full definition when the service fetches the run's catalog
items, so the fields are here, not an id to look up.
"""

import re
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from duo_workflow_service.agent_platform.v1.catalog.items.workspace_agent import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
)

__all__ = [
    "MAX_AI_CATALOG_NAME_LENGTH",
    "MAX_AI_CATALOG_PROMPT_LENGTH",
    "AiCatalogAgent",
]

MAX_AI_CATALOG_NAME_LENGTH = 255

# GitLab sets no length limit on a catalog agent's prompt. It limits only the whole
# definition, to DEFINITION_SIZE_LIMIT = 80.kilobytes (ee/app/models/ai/catalog/item_version.rb).
# This limit counts characters and GitLab counts bytes, so every prompt GitLab accepts passes.
MAX_AI_CATALOG_PROMPT_LENGTH = 80 * 1024

_NAME_PREFIX = "ai-catalog/agents/"
_CHARACTERS_A_COMPONENT_NAME_CANNOT_HOLD = re.compile(r"[|:#.]")

_Identifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_NAME_LENGTH),
]
_Name = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=MAX_AI_CATALOG_NAME_LENGTH
    ),
]
_Description = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=MAX_DESCRIPTION_LENGTH
    ),
]
_Prompt = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=MAX_AI_CATALOG_PROMPT_LENGTH
    ),
]


class AiCatalogAgent(BaseModel):
    """An agent published to the AI Catalog, attached as a subagent.

    ``catalog_name`` is the name in the AI Catalog. ``name`` is the component name inside the flow, the same shape
    ``WorkspaceAgent`` uses, so the two kinds cannot collide and a session log names the source.
    """

    item_id: _Identifier
    version: _Identifier
    catalog_name: _Name = Field(alias="name")
    description: _Description
    toolset: list[_Identifier] = Field(default_factory=list)
    prompt: _Prompt

    @property
    def name(self) -> str:
        readable = _CHARACTERS_A_COMPONENT_NAME_CANNOT_HOLD.sub("-", self.catalog_name)
        suffix = f" ({self.item_id})"
        room = MAX_AI_CATALOG_NAME_LENGTH - len(_NAME_PREFIX) - len(suffix)
        return f"{_NAME_PREFIX}{readable[:room]}{suffix}"
