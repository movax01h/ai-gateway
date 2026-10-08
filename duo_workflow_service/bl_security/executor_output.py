"""Reading the executor's text envelope: the exit-code header and the truncation notices.

The Node executor prefixes a command's output with an ``Exit code: N`` line and cuts output over 50 KB to its head and
tail around a marker line. Any action response over 4 MiB is replaced by a head+tail sample that starts with a notice
and still reports success. A truncated listing is not a complete list, so callers must detect it.
"""

import re

__all__ = ["EXIT_CODE_HEADER", "complete_lines", "executor_truncated"]

EXIT_CODE_HEADER = re.compile(r"\AExit code: (\S*)\n")
_OUTPUT_TRUNCATED = re.compile(r"^\[\.\.\. output truncated: .*\]$", re.MULTILINE)
_SIZE_LIMIT_NOTICE = re.compile(r"\AMaximum allowed size exceeded\. .*\n?")


def executor_truncated(text: str) -> bool:
    """Whether the executor cut ``text``: the size-limit notice or the truncation marker."""
    return bool(_SIZE_LIMIT_NOTICE.match(text) or _OUTPUT_TRUNCATED.search(text))


def complete_lines(text: str) -> list[str]:
    """The lines of a truncated listing that survived whole.

    Drops the notice, the marker, and the two lines the byte cut split.
    """
    lines = _SIZE_LIMIT_NOTICE.sub("", text, count=1).splitlines()
    for i, line in enumerate(lines):
        if _OUTPUT_TRUNCATED.match(line):
            return lines[: max(i - 1, 0)] + lines[i + 2 :]
    return lines
