"""Fetching each file's diff and turning it into the shared `<file_diff>` representation.

Formatted with the same helpers the merge-request path uses, so both surfaces produce the representation documented by
`common/diff_structure/1.0.0.jinja`.
"""

from typing import Dict, List, Tuple

from ai.features.review.code_review.components.diff_format import (
    format_file_diff,
    is_binary_diff,
    split_unified_diff,
)
from ai.features.review.code_review.components.local_review.exclusions import (
    EXCLUSION_NO_DIFF,
    EXCLUSION_NON_TEXT,
    EXCLUSION_OVERSIZED,
    EXCLUSION_TRUNCATED,
    EXCLUSION_UNREADABLE,
    MAX_DIFF_LINES_PER_FILE,
)
from ai.features.review.code_review.components.local_review.git_output import (
    FileChange,
    is_text,
)
from ai.features.review.code_review.components.local_review.target import ReviewTarget
from ai.features.review.code_review.components.local_review.transport import (
    DIFF_FLAGS,
    GitRunner,
)
from duo_workflow_service.bl_security.executor_output import executor_truncated


async def fetch_diffs(
    git: GitRunner,
    target: ReviewTarget,
    paths: List[str],
    numstat: Dict[str, FileChange],
    drivers: Tuple[str, ...] = (),
) -> Tuple[Dict[str, str], Dict[str, str], List[Tuple[str, str]]]:
    """Fetch one file's diff per call, and account for every path that does not come back.

    One batched call cannot work: the executor caps a command's output at roughly 50KB and
    truncates the middle, which drops whole files from a change of any size while the rest of
    the payload goes on listing them. Per file the cap is reachable only by a genuinely
    enormous single diff, and that case is reported rather than half-rendered.
    """
    blocks: Dict[str, str] = {}
    renamed: Dict[str, str] = {}
    excluded: List[Tuple[str, str]] = []

    for path in sorted(paths):
        # git only detects a rename when both sides are in the pathspec.
        old_path = numstat[path].old_path if path in numstat else path
        raw = await git.run(
            "diff",
            *DIFF_FLAGS,
            *target.range,
            "--",
            *sorted({path, old_path}),
            filter_drivers=drivers,
        )

        if executor_truncated(raw):
            excluded.append((path, EXCLUSION_TRUNCATED))
            continue

        diffs, detected = split_unified_diff(raw)
        # Emit only what selection approved, whatever git returned.
        diff = diffs.get(path)
        if diff is None:
            excluded.append((path, EXCLUSION_NO_DIFF))
            continue

        # `--numstat` calls a file text if it finds no NUL early, so only content is reliable.
        if is_binary_diff(diff) or not is_text(diff):
            excluded.append((path, EXCLUSION_NON_TEXT))
            continue

        blocks[path] = format_file_diff(path, diff)
        if path in detected:
            renamed[path] = detected[path]

    return (
        dict(sorted(blocks.items())),
        renamed,
        sorted(excluded),
    )


async def fetch_untracked_diffs(
    git: GitRunner, allowed: List[str]
) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """Read each untracked file whole as one all-added hunk, excluding unreadable, non-text and oversized ones."""
    excluded: List[Tuple[str, str]] = []
    blocks: Dict[str, str] = {}
    for path in allowed:
        content = await git.read_file(path)
        if content is None:
            excluded.append((path, EXCLUSION_UNREADABLE))
            continue
        if not is_text(content):
            excluded.append((path, EXCLUSION_NON_TEXT))
            continue

        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if len(lines) > MAX_DIFF_LINES_PER_FILE:
            excluded.append((path, EXCLUSION_OVERSIZED))
            continue

        diff = "@@ -0,0 +1,{count} @@\n{body}".format(
            count=len(lines), body="\n".join(f"+{line}" for line in lines)
        )
        blocks[path] = format_file_diff(path, diff)

    return blocks, sorted(excluded)
