"""Unit tests for reading the executor's text envelope."""

import pytest

from duo_workflow_service.bl_security.executor_output import (
    EXIT_CODE_HEADER,
    complete_lines,
    executor_truncated,
)

MARKER = "[... output truncated: showing first and last 25KB of 80KB ...]"
NOTICE = "Maximum allowed size exceeded. Result: 9 bytes. Maximum allowed: 8 bytes\n"


def test_the_exit_code_header_is_read_and_anchored():
    header = EXIT_CODE_HEADER.match("Exit code: 2\n./a.rb\n")
    assert header and header.group(1) == "2"
    assert EXIT_CODE_HEADER.match("./a.rb\nExit code: 2\n") is None


@pytest.mark.parametrize(
    ("text", "truncated"),
    [
        ("a.rb\nb.rb\n", False),
        (f"a.rb\n{MARKER}\nz.rb\n", True),
        (NOTICE + "a.rb\n", True),
        # The notice only counts at the start of the response.
        ("a.rb\n" + NOTICE, False),
    ],
)
def test_truncation_is_detected(text, truncated):
    assert executor_truncated(text) is truncated


def test_complete_lines_drop_the_marker_and_the_two_split_paths():
    text = "a.rb\nb.r\n" + MARKER + "\n.rb\nz.rb"
    assert complete_lines(text) == ["a.rb", "z.rb"]


def test_a_notice_without_a_marker_drops_only_the_notice():
    assert complete_lines(NOTICE + "a.rb") == ["a.rb"]


def test_an_untruncated_listing_is_every_line():
    assert complete_lines("a.rb\nb.rb") == ["a.rb", "b.rb"]
