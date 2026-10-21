"""Shared formatting of git diffs into the `<file_diff>`/`<line>` representation.

The format is documented to the model by `ai_gateway/prompts/definitions/common/diff_structure/1.0.0.jinja`, so any
change here must be mirrored there. Holding it in one place is what keeps every caller rendering it identically; the
merge-request review path is the only caller today, and a second one arrives with local review.
"""

import re
from typing import Dict, Iterator, Optional, Set, Tuple

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
    return "Binary files" in raw_diff


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

    return f'<file_diff filename="{file_path}">\n{formatted}\n</file_diff>'


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
        formatted.append(f'<file old_path="{old_path}" new_path="{new_path}"></file>')
    formatted.append("</renamed_files>")

    return "\n".join(formatted)
