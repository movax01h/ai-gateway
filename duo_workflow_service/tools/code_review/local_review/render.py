"""The `<input>` payload's tags, other than `<review_target>`.

There is no title or description
--------------------------------
The merge-request payload opens with `<mr_title>` and `<mr_description>`. Neither has a local
equivalent, and neither is emitted: an empty MR-flavoured tag would tell the reviewer it has the
author's stated intent when it has nothing. What replaces them is `<review_target>`, which says
what is being compared, and the optional `<review_intent>`, rendered only when a caller supplies
one.

`intent` is deliberately not defaulted from the session goal. In an interactive session the goal
is typically a reference -- "implement <issue url>" -- so defaulting to it would put a URL where
the reviewer expects a rationale. A caller that has the real intent passes it; one that does not
omits the tag.
"""

import html
from typing import Dict, List, Optional, Tuple

from duo_workflow_service.tools.code_review.diff_format import format_renamed_files
from duo_workflow_service.tools.code_review.local_review.git_output import (
    WorkingTreeStatus,
)

_EMPTY_DIFF = "<empty_diff>true</empty_diff>"
_ALL_EXCLUDED = (
    "<all_files_excluded>\n"
    "Every file in scope was excluded, so there is no diff to review. Report that, and why, rather "
    "than reporting that nothing changed.\n"
    "</all_files_excluded>"
)


def escape_text(value: str) -> str:
    """Escape a path for element text, where only `<` can end the enclosing construct.

    Narrower than the attribute case, which must also handle `"`. Nothing downstream decodes
    entities -- the reviewer echoes these paths back to anchor a finding and to open files --
    so anything escaped beyond what can forge a tag is a path it will then get wrong.
    """
    return value.replace("<", "&lt;")


def render_intent(intent: Optional[str]) -> str:
    """The local stand-in for <mr_description>, escaped as ambient escapes that field."""
    if not intent or not intent.strip():
        return ""

    return f"<review_intent>\n{html.escape(intent.strip())}\n</review_intent>"


def render_list(tag: str, values: List[str]) -> str:
    """`<tag>` holding one `- value` line per entry, escaped; nothing at all when there are none."""
    if not values:
        return ""

    body = "\n".join(f"- {escape_text(value)}" for value in values)

    return f"<{tag}>\n{body}\n</{tag}>"


def render_pairs(tag: str, pairs: List[Tuple[str, str]]) -> str:
    """`<tag>` holding one `- path: reason` line per pair, the path escaped; nothing at all when there are none."""
    if not pairs:
        return ""

    body = "\n".join(f"- {escape_text(path)}: {reason}" for path, reason in pairs)

    return f"<{tag}>\n{body}\n</{tag}>"


def render_uncommitted(mode: str, status: WorkingTreeStatus) -> str:
    """Never a silently partial review.

    In `branch` mode the review runs over the committed diff, so every file with uncommitted or
    untracked changes is named here as out of scope. In `working_tree` mode those changes are the
    target, and nothing is rendered.
    """
    if mode != "branch" or not status.is_dirty:
        return ""

    body = "\n".join(
        f"- {escape_text(path)}" for path in [*status.modified, *status.untracked]
    )

    return (
        "<uncommitted_changes>\n"
        "These files have uncommitted changes, which this review does not cover because it is "
        "restricted to committed work. Do not report findings about those uncommitted changes. Where a "
        "file also appears in <changed_files>, review only the lines shown in its diff.\n"
        f"{body}\n"
        "</uncommitted_changes>"
    )


def render_body(
    blocks: Dict[str, str], renamed: Dict[str, str], excluded: List[Tuple[str, str]]
) -> str:
    """Distinguish "nothing changed" from "everything was excluded".

    Both leave no diff behind, but only the first means there is nothing to review. Conflating them turns a large-but-
    legitimate change into a silent no-op review.
    """
    if blocks:
        return _render_diffs(blocks, renamed)

    return _ALL_EXCLUDED if excluded else _EMPTY_DIFF


def _render_diffs(blocks: Dict[str, str], renamed: Dict[str, str]) -> str:
    sections = ["\n\n".join(blocks.values()), format_renamed_files(renamed)]
    body = "\n\n".join(section for section in sections if section)

    return f"<git_diffs>\n{body}\n</git_diffs>"
