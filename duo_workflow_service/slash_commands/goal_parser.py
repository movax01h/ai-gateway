from typing import Optional, Tuple


def parse(goal: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse a goal string to extract the slash command and remaining text.

    Args:
        goal: The user input string (e.g., "/explain This code is confusing")

    Returns:
        ParsedSlashCommand containing the command type and remaining text
    """
    goal = goal.strip()

    parts = goal.split(maxsplit=1)

    return _parse_parts(parts)


def _parse_parts(parts: list) -> Tuple[Optional[str], Optional[str]]:
    """Parse command parts to extract command type and remaining text.

    Args:
        parts (list): List of command parts split by some delimiter

    Returns:
        tuple: (command_type, remaining_text) where remaining_text may be None
    """
    command_part = parts[0][1:]

    if command_part == "":
        command_type, remaining_text = _parse_space_after_slash(parts)
    else:
        command_type = command_part
        remaining_text = parts[1] if len(parts) > 1 else None

    return command_type, remaining_text


def _parse_space_after_slash(parts: list) -> Tuple[Optional[str], Optional[str]]:
    """Parse command parts to extract command type and remaining text if there is a space after the slash.

    Args:
        parts (list): List of command parts

    Returns:
        tuple: (command_type, remaining_text) where remaining_text may be None
    """
    if len(parts) <= 1:
        return None, None

    remaining_parts = parts[1].split(maxsplit=1)
    command_type = remaining_parts[0]
    remaining_text = remaining_parts[1] if len(remaining_parts) > 1 else None

    return command_type, remaining_text


def is_slash_command(goal: str) -> bool:
    """Determine if the input text is formatted like a slash command.

    Args:
        goal: The user input string

    Returns:
        bool: True if the text is formatted like a slash command, False otherwise
    """
    content = goal.strip()
    if not content.startswith("/"):
        return False

    # Exclude file paths like '/home/dir'
    if content.split(" ")[0].count("/") > 1:
        return False

    return True
