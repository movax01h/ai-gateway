"""Feature-owned tools for the duo_permissions_assistant flow."""

from langchain.tools import BaseTool

from .set_form_permissions import SetFormPermissions
from .update_form_permissions import UpdateFormPermissions

__all__ = ["FEATURE_TOOLS", "SetFormPermissions", "UpdateFormPermissions"]

# Tools this feature owns, grouped by the agent privilege that gates them.
# The tools registry merges these into its privilege map at discovery time.
FEATURE_TOOLS: dict[str, list[type[BaseTool]]] = {
    "read_write_gitlab": [UpdateFormPermissions, SetFormPermissions],
}
