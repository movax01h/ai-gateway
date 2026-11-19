"""The prompt a component names, as a flow author writes it in the component's YAML entry.

`prompt_id` and `prompt_version` are parameters of every component that runs a prompt, so they belong to the Flow
Registry's public contract: renaming or retyping either one is a breaking change for stable versions (see
`docs/flow_registry/contribution_guidelines.md`). They are declared here, in the component layer, and not in the prompt
module, so a refactor of the prompt module can't change that contract without touching this file.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = ["PromptReference", "prompt_reference_of"]


class PromptReference(BaseModel):
    """The two fields a component declares to name the prompt it runs.

    A component inherits it to get the fields declared once. It is not frozen, because a component that inherits it is
    mutated after it is built.
    """

    prompt_id: str
    # Unset means the prompt is declared inline in the flow. Set, it names a stored prompt file's version.
    prompt_version: str | None = None

    @property
    def is_stored(self) -> bool:
        """Whether this names a stored prompt file, rather than one declared inline in the flow."""
        return bool(self.prompt_version)


class _ReadPromptReference(PromptReference):
    """Reads a reference out of an arbitrary mapping, which carries fields it doesn't know."""

    model_config = ConfigDict(extra="ignore")


def prompt_reference_of(declaration: Mapping[str, Any]) -> PromptReference | None:
    """Read the prompt a component entry names, without knowing which component it is.

    Args:
        declaration: A component entry as the flow config holds it. It may belong to a component that runs no
            prompt, such as human input.

    Returns:
        The reference, or None when the entry names no prompt.

    Raises:
        ValidationError: The entry names a version and no prompt, or a field isn't a string.
    """
    if not declaration.get("prompt_id") and not declaration.get("prompt_version"):
        return None

    return _ReadPromptReference.model_validate(declaration)
