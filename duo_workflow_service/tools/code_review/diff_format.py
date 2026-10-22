"""Shared formatting of git diffs into the `<file_diff>`/`<line>` representation.

The format is documented to the model by `ai_gateway/prompts/definitions/common/diff_structure/1.0.0.jinja`, so any
change here must be mirrored there. Holding it in one place is what keeps every caller rendering it identically:
the merge-request path and the local checkout path both render through these helpers.
"""

import re
from typing import Dict, Iterator, Optional, Set, Tuple

_DIFF_HEADER_PREFIX = "diff --git "
_BINARY_RE = re.compile(r"^(Binary files |GIT binary patch)", re.MULTILINE)
_HEADER_PREFIXES = (
    "index ",
    "old mode ",
    "new mode ",
    "new file mode ",
    "deleted file mode ",
    "similarity index ",
    "dissimilarity index ",
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
)


def is_binary_diff(raw_diff: str) -> bool:
    """Whether git marked this diff binary, by either of the two markers it emits.

    Anchored to the start of a line because every content line in a diff is prefixed with `+`, `-` or a space, so only
    git can begin one with these words. Matching anywhere drops any file whose content merely mentions the phrase -- a
    test fixture, a parser, this module's own tests.
    """
    return bool(_BINARY_RE.search(raw_diff))


def walk_diff_lines(raw_diff: str) -> Iterator[tuple[str, int, int, str]]:
    """Walk a unified diff, yielding `(kind, old_line, new_line, text)` per line.

    `kind` is one of `chunk_header`, `nonewline`, `added`, `deleted` or
    `context`. The line numbers belong to the yielded line, not the running
    counters. File metadata lines are skipped.

    Metadata is only skipped before the first hunk header. Inside a hunk a `+++`
    is an added line whose own text starts with `++`, and a `---` is a deleted
    line starting `--`. Skipping those would drop the line and leave every later
    counter in the hunk one short, which misplaces the rendered line number and
    stops the incremental-diff collector matching the same line across two diffs.
    """
    line_old = 1
    line_new = 1
    in_hunk = False

    for line in raw_diff.split("\n"):
        if not line:
            continue

        if line.startswith("@@"):
            # Parse chunk header
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if match:
                line_old = int(match.group(1))
                line_new = int(match.group(2))
                in_hunk = True
                yield "chunk_header", line_old, line_new, line
            continue

        # A new file section closes the hunk that came before it, so its own
        # `---`/`+++` headers are metadata again.
        if line.startswith("diff --git"):
            in_hunk = False
            continue

        # Local diffs carry full file headers; API diffs start at the first hunk and never
        # reach this. Only `+++`/`---` are ambiguous, hence the in-hunk guard on those.
        if not in_hunk and (
            line.startswith(("+++", "---")) or line.startswith(_HEADER_PREFIXES)
        ):
            continue

        # Handle "No newline at end of file"
        if line.startswith("\\"):
            yield "nonewline", line_old, line_new, line
            continue

        # Determine line type and extract text without prefix
        if line.startswith("+"):
            yield "added", line_old, line_new, line[1:]
            line_new += 1
        elif line.startswith("-"):
            yield "deleted", line_old, line_new, line[1:]
            line_old += 1
        elif line.startswith(" "):
            yield "context", line_old, line_new, line[1:]
            line_old += 1
            line_new += 1
        else:
            # Unexpected line format, treat as context
            yield "context", line_old, line_new, line
            line_old += 1
            line_new += 1


def format_diff_lines(
    raw_diff: str,
    file_path: Optional[str] = None,
    changed_lines: Optional[Set[Tuple[str, int]]] = None,
) -> str:
    """Format each line of a single file's diff with its type and line numbers.

    Added lines listed in `changed_lines` are marked with `since_last_review="true"`.
    """
    if not raw_diff.strip() or is_binary_diff(raw_diff):
        return ""

    lines = []
    for kind, line_old, line_new, text in walk_diff_lines(raw_diff):
        if kind == "chunk_header":
            lines.append(f"<chunk_header>{text}</chunk_header>")
            continue

        # An added line has no old number and a deleted line has no new one.
        # A context or nonewline line carries both.
        old = "" if kind == "added" else line_old
        new = "" if kind == "deleted" else line_new
        marked = (
            kind == "added" and changed_lines and (file_path, line_new) in changed_lines
        )
        marker = ' since_last_review="true"' if marked else ""

        lines.append(
            f'<line type="{kind}" old_line="{old}" new_line="{new}"{marker}>{text}</line>'
        )

    return "\n".join(lines)


def format_file_diff(
    file_path: str,
    raw_diff: str,
    changed_lines: Optional[Set[Tuple[str, int]]] = None,
) -> str:
    """Wrap one file's formatted diff in a `<file_diff>` element."""
    formatted = format_diff_lines(raw_diff, file_path, changed_lines)

    return f'<file_diff filename="{_attribute(file_path)}">\n{formatted}\n</file_diff>'


def format_file_diffs(
    diffs_and_paths: Dict[str, str],
    changed_lines: Optional[Set[Tuple[str, int]]] = None,
) -> str:
    """Wrap each formatted file diff in a `<file_diff>` element."""
    return "\n\n".join(
        format_file_diff(file_path, diff_content, changed_lines)
        for file_path, diff_content in diffs_and_paths.items()
    )


def format_renamed_files(renamed_files: Dict[str, str]) -> str:
    """Render a `<renamed_files>` section mapping new paths back to old paths."""
    if not renamed_files:
        return ""

    formatted = ["<renamed_files>"]
    for new_path, old_path in renamed_files.items():
        formatted.append(
            f'<file old_path="{_attribute(old_path)}" '
            f'new_path="{_attribute(new_path)}"></file>'
        )
    formatted.append("</renamed_files>")

    return "\n".join(formatted)


def _attribute(path: str) -> str:
    """Escape a path for use as an XML attribute value.

    Paths are the one part of this representation an attacker picks freely: a directory `<` holding
    a file `file_diff>` is a legal path spelling `</file_diff>`, which would close the element
    early and let the rest of the payload be read as a forged one. The prompt-security scanner does
    not cover it -- it encodes `<goal>` and `<system>`, not the tags this format defines.

    Only `"` and `<` are escaped. Nothing downstream decodes entities -- the reviewer reads the
    attribute as text and echoes the path back to anchor a finding -- so every entity emitted is
    a path it will then get wrong. `&` and `'` are left alone because they are common in real
    paths and neither can end the attribute; the cost is that a path spelling `&quot;` reads as
    one containing `"`.
    """
    return path.replace('"', "&quot;").replace("<", "&lt;")


def split_unified_diff(raw: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Split multi-file `git diff` output into per-file diffs plus detected renames.

    Returns `({path: raw_diff}, {new_path: old_path})`, matching the shapes the merge-request path already builds from
    the API response.
    """
    diffs: Dict[str, str] = {}
    renamed: Dict[str, str] = {}

    for chunk in _iter_file_chunks(raw):
        old_path, new_path = _chunk_paths(chunk)
        path = new_path or old_path
        if not path:
            continue

        diffs[path] = chunk
        if old_path and new_path and old_path != new_path:
            renamed[new_path] = old_path

    return diffs, renamed


def _iter_file_chunks(raw: str) -> Iterator[str]:
    chunk: list[str] = []
    for line in raw.split("\n"):
        if line.startswith(_DIFF_HEADER_PREFIX):
            if chunk:
                yield "\n".join(chunk)
            chunk = [line]
        elif chunk:
            chunk.append(line)

    if chunk:
        yield "\n".join(chunk)


def _chunk_paths(chunk: str) -> Tuple[Optional[str], Optional[str]]:
    old_path: Optional[str] = None
    new_path: Optional[str] = None

    for line in chunk.split("\n"):
        if line.startswith("@@"):
            break
        if line.startswith("rename from "):
            old_path = _unquote_git_path(line[len("rename from ") :])
        elif line.startswith("rename to "):
            new_path = _unquote_git_path(line[len("rename to ") :])
        elif line.startswith("--- "):
            old_path = _strip_diff_prefix(line[4:], "a/")
        elif line.startswith("+++ "):
            new_path = _strip_diff_prefix(line[4:], "b/")
            break

    if old_path is None and new_path is None:
        return _paths_from_header(chunk.split("\n", 1)[0])

    return old_path, new_path


def _strip_diff_prefix(value: str, prefix: str) -> Optional[str]:
    # A filename's own leading and trailing spaces are data; the tab git appends here to
    # delimit them is not. A real tab in a name would force quoting, so this cannot eat one.
    path = _unquote_git_path(value.rstrip("\t"))
    if path == "/dev/null":
        return None

    return path.removeprefix(prefix)


def _unquote_git_path(value: str) -> str:
    if not (value.startswith('"') and value.endswith('"') and len(value) > 1):
        return value

    unescaped = (
        value[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape")
    )
    # core.quotePath octal-escapes raw UTF-8 bytes, not codepoints, so reassemble and decode them.
    try:
        return unescaped.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return unescaped


def _paths_from_header(header: str) -> Tuple[Optional[str], Optional[str]]:
    rest = header[len(_DIFF_HEADER_PREFIX) :].rstrip("\n")
    if not rest:
        return None, None

    if rest.startswith('"'):
        closing = rest.find('" "', 1)
        if closing == -1:
            return None, None
        return (
            _strip_diff_prefix(rest[: closing + 1], "a/"),
            _strip_diff_prefix(rest[closing + 2 :], "b/"),
        )

    # Unprefixed paths may contain spaces, making `a/<old> b/<new>` ambiguous. The
    # identical-path split is the only unambiguous reading, and it covers the cases
    # that reach here (mode changes and binary files, which git emits without
    # `---`/`+++` lines).
    midpoint = (len(rest) - 1) // 2
    if rest[midpoint] == " " and rest[:midpoint] == f"a/{rest[midpoint + 3 :]}":
        path = rest[midpoint + 3 :]
        return path, path

    _, _, tail = rest.partition(" b/")

    return None, tail or None
