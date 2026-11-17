from typing import List, Type

from duo_workflow_service.security.secret_redaction import REDACTED_PLACEHOLDER
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


def get_all_subclasses(cls):
    all_subclasses = set()

    for subclass in cls.__subclasses__():
        all_subclasses.add(subclass)
        all_subclasses.update(get_all_subclasses(subclass))

    return all_subclasses


def get_all_ops_tools(tool_class: type = DuoBaseTool) -> List[Type[DuoBaseTool]]:
    tool_classes = []

    for subclass in get_all_subclasses(tool_class):
        # Check if the subclass has _execute method
        if hasattr(subclass, "_execute"):
            method = getattr(subclass, "_execute")

            # Check if it's not abstract
            if not getattr(method, "__isabstractmethod__", False):
                tool_classes.append(subclass)

    return tool_classes


def added_redactions(written: str, replaced: str) -> int:
    """Count the redaction placeholders a write adds beyond those it replaces.

    Args:
        written: The text the tool is about to write.
        replaced: The text it replaces. Pass an empty string when there is no baseline.

    Returns:
        How many more placeholders `written` holds than `replaced`, never below zero.
    """
    return max(
        written.count(REDACTED_PLACEHOLDER) - replaced.count(REDACTED_PLACEHOLDER), 0
    )


def adds_redactions(written: str, replaced: str) -> bool:
    """Report whether a write adds more redaction placeholders than it replaces.

    Redaction swaps a secret for the placeholder in what the model reads, so writing it back would overwrite the
    real value. Only the count of placeholders is compared, so a write that keeps the existing ones adds none.
    The caller logs and raises the refusal, so its message can advise the model about that tool.

    Args:
        written: The text the tool is about to write.
        replaced: The text it replaces. Pass an empty string when there is no baseline.

    Returns:
        True when `written` holds more placeholders than `replaced`.
    """
    return added_redactions(written, replaced) > 0
