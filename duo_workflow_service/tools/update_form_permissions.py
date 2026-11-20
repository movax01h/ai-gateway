"""Deprecated import location. Moved into the duo_permissions_assistant feature package.

Import from ``ai.features.chat.duo_permissions_assistant.components`` instead.
"""

import warnings

from ai.features.chat.duo_permissions_assistant.components.update_form_permissions import (
    UpdateFormPermissions,
    UpdateFormPermissionsInput,
)

warnings.warn(
    "duo_workflow_service.tools.update_form_permissions has moved to "
    "ai.features.chat.duo_permissions_assistant.components.update_form_permissions",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["UpdateFormPermissions", "UpdateFormPermissionsInput"]
