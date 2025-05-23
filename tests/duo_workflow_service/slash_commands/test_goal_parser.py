# pylint: disable=file-naming-for-tests

import pytest

from duo_workflow_service.slash_commands.goal_parser import parse


@pytest.mark.parametrize(
    "goal,expected_command,expected_text",
    [
        (
            "/explain def add(a, b): return a + b",
            "explain",
            "def add(a, b): return a + b",
        ),
        ("/search GitLab API", "search", "GitLab API"),
        ("/help", "help", None),
        ("/invalid-command", "invalid-command", None),
        ("/ space-at-start", "space-at-start", None),
        (
            "/   space-at-start and more after commands space",
            "space-at-start",
            "and more after commands space",
        ),
        (
            "/multiple   spaces   between   words",
            "multiple",
            "spaces   between   words",
        ),
        ("/trim-spaces    ", "trim-spaces", None),
        ("   /leading-whitespace", "leading-whitespace", None),
        (
            "/explain why this code isn't testable/maintainable",
            "explain",
            "why this code isn't testable/maintainable",
        ),
        ("/", None, None),
    ],
)
def test_parse_slash_commands(goal, expected_command, expected_text):
    """
    Test that parse correctly extracts command types and remaining text from various slash command formats.
    This parametrized test covers multiple command formats including commands with arguments,
    commands without arguments, commands with extra spaces, and edge cases.
    The parse method now returns a tuple of (command_type, remaining_text) instead of a class instance.
    """
    command_type, remaining_text = parse(goal)
    assert command_type == expected_command
    assert remaining_text == expected_text
