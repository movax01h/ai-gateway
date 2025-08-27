import re
from typing import Optional

_QUICK_ACTION_REGEX = re.compile(r"(?mi)^\s*/[a-z][a-z_]*(?=\s|$)")


def validate_no_quick_actions(
    text: Optional[str], *, field: str = "text"
) -> Optional[str]:
    """Return an error message if `text` contains GitLab quick actions; otherwise None."""
    if not text:
        return None
    if _QUICK_ACTION_REGEX.search(text):
        return (
            f"{field.capitalize()} contains GitLab quick actions, which are not allowed. "
            "Quick actions are commands on their own line starting with '/'. "
            "Examples include /merge, /approve, /close.To include it literally, "
            "escape the slash (\\/) or add a non-whitespace char before it."
        )
    return None
