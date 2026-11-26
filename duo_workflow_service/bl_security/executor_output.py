"""Reading the executor's text envelope: the exit-code header and the truncation notices.

The Node executor prefixes a command's output with an ``Exit code: N`` line and cuts output over 50 KB to its head and
tail around a marker line. Any action response over 4 MiB is replaced by a head+tail sample that starts with a notice
and still reports success. A truncated listing is not a complete list, so callers must detect it.
"""

import re
from typing import NamedTuple, Optional

__all__ = [
    "EXIT_CODE_HEADER",
    "CommandOutput",
    "complete_lines",
    "executor_truncated",
    "strip_size_limit_notice",
    "unwrap_command_output",
]

EXIT_CODE_HEADER = re.compile(r"\AExit code: (\S*)\n")
_OUTPUT_TRUNCATED = re.compile(r"^\[\.\.\. output truncated: .*\]$", re.MULTILINE)
_SIZE_LIMIT_NOTICE = re.compile(r"\AMaximum allowed size exceeded\. .*\n?")


def executor_truncated(text: str) -> bool:
    """Whether the executor cut ``text``: the size-limit notice or the truncation marker."""
    return bool(_SIZE_LIMIT_NOTICE.match(text) or _OUTPUT_TRUNCATED.search(text))


def strip_size_limit_notice(text: str) -> str:
    """``text`` without the leading size-limit notice, if it carries one."""
    return _SIZE_LIMIT_NOTICE.sub("", text, count=1)


class CommandOutput(NamedTuple):
    """A command response split at its envelope; ``code`` is ``None`` when it came bare."""

    code: Optional[str]
    body: str


def unwrap_command_output(raw: str) -> CommandOutput:
    """Split a ``runCommand`` response into the exit code it reports and the output after it.

    The action reports success whatever the command exited with, so the header is the only signal of a failure; what
    counts as one is the caller's to decide, and nothing is raised here. The code stays text, because the header
    admits any non-space token. A response over the executor's size limit puts its notice *before* the header; the
    notice is stripped to find the header and kept on the body, so ``executor_truncated`` still sees it. An executor
    that returns the output bare is tolerated: the response comes back whole, with no code.
    """
    text = strip_size_limit_notice(raw)
    header = EXIT_CODE_HEADER.match(text)
    if header is None:
        return CommandOutput(None, raw)
    notice = raw[: len(raw) - len(text)]
    return CommandOutput(header.group(1), notice + text[header.end() :])


def complete_lines(text: str) -> list[str]:
    """The lines of a truncated listing that survived whole.

    Drops the notice, the marker, and the two lines the byte cut split.
    """
    lines = strip_size_limit_notice(text).splitlines()
    for i, line in enumerate(lines):
        if _OUTPUT_TRUNCATED.match(line):
            return lines[: max(i - 1, 0)] + lines[i + 2 :]
    return lines
