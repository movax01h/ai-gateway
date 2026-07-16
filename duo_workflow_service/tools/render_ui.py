import json
from pathlib import Path
from typing import Any, ClassVar, Dict, Type

import jsonschema
from langchain_core.tools import ToolException
from packaging.version import Version
from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

__all__ = ["UI_TREE_JSON_SCHEMA", "RenderUiTool"]

# SSOT: the declarative `ui_tree` JSON Schema is generated from the @gitlab/duo-ui
# gen-UI catalog (`uiTreeJsonSchema`) and vendored here. It constrains the agent's
# `ui_tree` to known catalog components with valid props — the same schema the
# frontend validates against (`validateUiTree`), so the contracts cannot drift.
# Regenerate when the catalog changes; automation is tracked in
# gitlab-org/gitlab#604060.
_UI_TREE_SCHEMA_PATH = Path(__file__).parent / "render_ui_schema.json"
UI_TREE_JSON_SCHEMA: Dict[str, Any] = json.loads(_UI_TREE_SCHEMA_PATH.read_text())


class RenderUiInput(BaseModel):
    """Input schema for ``RenderUiTool``: a declarative ``ui_tree``."""

    tree: Dict[str, Any] = Field(
        description=(
            "A declarative ui_tree of Duo Chat gen-UI catalog components: an "
            "object with a `root` node id and an `elements` map of id -> "
            "{ type, props, children? }. `type` MUST be a catalog component and "
            "`props` MUST match that component's schema."
        ),
        json_schema_extra=UI_TREE_JSON_SCHEMA,
    )


class RenderUiTool(DuoBaseTool):
    """Render a declarative UI (a ``ui_tree`` of catalog components) to the user."""

    name: str = "render_ui"
    description: str = """Render a declarative UI by composing gen-UI catalog components into a ui_tree.

    Use this when a structured, interactive presentation is clearer than plain
    text. The `tree` MUST conform to the Duo Chat gen-UI catalog: every node's
    `type` must be a known catalog component and its `props` must match that
    component's schema. Unknown components or invalid props are rejected — fix
    and retry using only components from the catalog.
    """
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL
    args_schema: Type[BaseModel] = RenderUiInput
    # < 1.0.0 keeps this internal-framework tool out of the AI Catalog (see
    # STABLE_VERSION_THRESHOLD in server.py); gen-UI is at an early stage.
    tool_version: ClassVar[Version] = Version("0.0.1")

    async def _execute(self, tree: Dict[str, Any], **_kwargs: Any) -> str:
        # Validate against the catalog SSOT schema; on failure raise ToolException
        # with a corrective message so the agent can fix the tree and retry
        # (constrained to the catalog, no hallucinated components/props).
        try:
            jsonschema.validate(instance=tree, schema=UI_TREE_JSON_SCHEMA)
        except jsonschema.ValidationError as error:
            raise ToolException(f"Invalid ui_tree: {error.message}") from error

        # JSON Schema can't express the cross-property constraint that `root`
        # references an existing node — check it explicitly (matches the frontend
        # validateUiTree).
        root = tree.get("root")
        if root not in tree.get("elements", {}):
            raise ToolException(f'Invalid ui_tree: root "{root}" is not in elements.')

        return json.dumps({"ui_tree": tree})

    def format_display_message(
        self, args: RenderUiInput, _tool_response: Any = None
    ) -> str:
        elements = args.tree.get("elements", {}) if isinstance(args.tree, dict) else {}
        count = len(elements)
        return f"Rendering UI ({count} component{'' if count == 1 else 's'})"
