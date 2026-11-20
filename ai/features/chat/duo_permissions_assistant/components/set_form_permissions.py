import json
from typing import Any, Type

from pydantic import BaseModel, Field, field_validator

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

# The access token form groups permissions into three access levels (boundaries).
SCOPE_KEYS = ("namespace", "user", "global")


def _normalize_buckets(value: Any) -> dict[str, list[str]]:
    """Keep only the known access levels, drop non-string/empty entries, and de-duplicate names within each level
    (preserving order).

    Keys and names are trimmed and access-level keys are lower-cased, so plausible model output like "Global" or "
    namespace" still lands in the right bucket. Anything the model emits under an unexpected key is discarded rather
    than forwarded to the form.
    """
    if not isinstance(value, dict):
        return {}

    by_scope = {str(key).strip().lower(): names for key, names in value.items()}
    result: dict[str, list[str]] = {}
    for scope in SCOPE_KEYS:
        names = by_scope.get(scope)
        if not isinstance(names, list):
            continue
        deduped = list(
            dict.fromkeys(
                name.strip() for name in names if isinstance(name, str) and name.strip()
            )
        )
        if deduped:
            result[scope] = deduped
    return result


class SetFormPermissionsInput(BaseModel):
    select: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Permissions to select, grouped by access level. An object whose only "
            'keys are "namespace", "user", and "global", each mapping to the list '
            "of permission names to select at that level. Map any GROUP or PROJECT "
            'permission to "namespace". Put a permission under multiple levels when '
            "the request spans them (e.g. snippets owned by the user AND snippets in "
            "groups/projects)."
        ),
    )
    clear: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Permissions to clear, grouped by access level, in the same shape as "
            '"select".'
        ),
    )

    @field_validator("select", "clear", mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> dict[str, list[str]]:
        return _normalize_buckets(value)


class SetFormPermissions(DuoBaseTool):
    """Tool that updates the access token form with suggested permissions grouped by access level.

    Returns a JSON payload for the frontend to apply; performs no GitLab write operations.
    """

    name: str = "set_form_permissions"
    description: str = (
        "Update the access token form with the suggested permissions, grouped by "
        "access level (boundary). Use this tool to select or clear fine-grained "
        'permissions, placing each under the correct level ("namespace", "user", '
        'or "global").'
    )
    args_schema: Type[BaseModel] = SetFormPermissionsInput

    async def _execute(
        self,
        select: dict[str, list[str]],
        clear: dict[str, list[str]],
    ) -> str:
        return json.dumps({"select": select, "clear": clear})

    def format_display_message(
        self, args: SetFormPermissionsInput, _tool_response: Any = None
    ) -> str:
        parts = []
        select = self._format_buckets(args.select)
        clear = self._format_buckets(args.clear)
        if select:
            parts.append(f"Select: {select}")
        if clear:
            parts.append(f"Clear: {clear}")
        if not parts:
            return "No permission changes"
        return "Update access token permissions — " + "; ".join(parts)

    @staticmethod
    def _format_buckets(buckets: dict[str, list[str]]) -> str:
        """Render the buckets as a human-readable "name (level, ...)" summary for the activity log.

        Permissions are grouped by name so one spanning multiple levels reads as a single entry, with levels ordered
        consistently by access level.
        """
        labels: dict[str, list[str]] = {}
        for scope in SCOPE_KEYS:
            for name in buckets.get(scope, []):
                labels.setdefault(name, []).append(scope)
        return ", ".join(
            f"{name} ({', '.join(scopes)})" for name, scopes in labels.items()
        )
