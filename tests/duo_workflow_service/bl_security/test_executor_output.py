"""Unit tests for reading the executor's text envelope."""

import pytest

from duo_workflow_service.bl_security.executor_output import (
    EXIT_CODE_HEADER,
    complete_lines,
    executor_truncated,
    strip_size_limit_notice,
    unwrap_command_output,
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (NOTICE + "Exit code: 0\na.rb\n", "Exit code: 0\na.rb\n"),
        ("Exit code: 0\na.rb\n", "Exit code: 0\na.rb\n"),
        ("a.rb\n" + NOTICE, "a.rb\n" + NOTICE),
    ],
)
def test_the_notice_is_stripped_only_from_the_start(text, expected):
    assert strip_size_limit_notice(text) == expected


@pytest.mark.parametrize(
    ("raw", "code", "body"),
    [
        ("Exit code: 0\na.rb\n", "0", "a.rb\n"),
        ("Exit code: 128\nfatal: boom\n", "128", "fatal: boom\n"),
        ("Exit code: 1\n", "1", ""),
        # A signal leaves no exit code; the header admits any non-space token, or an empty one.
        ("Exit code: null\nfatal: boom\n", "null", "fatal: boom\n"),
        ("Exit code: -13\nfatal: boom\n", "-13", "fatal: boom\n"),
        ("Exit code: \nfatal: boom\n", "", "fatal: boom\n"),
        # The notice precedes the header; it is read through and kept on the body.
        (NOTICE + "Exit code: 0\na.rb\n", "0", NOTICE + "a.rb\n"),
        (NOTICE + "Exit code: 128\nfatal: boom\n", "128", NOTICE + "fatal: boom\n"),
        # A bare response comes back whole, with no code.
        ("a.rb\n", None, "a.rb\n"),
        (NOTICE + "a.rb\n", None, NOTICE + "a.rb\n"),
        ("", None, ""),
        # The header only counts at the start of the response.
        ("a.rb\nExit code: 2\n", None, "a.rb\nExit code: 2\n"),
    ],
)
def test_a_command_response_is_split_at_its_envelope(raw, code, body):
    output = unwrap_command_output(raw)

    assert output == (code, body)
    assert executor_truncated(output.body) is executor_truncated(raw)
