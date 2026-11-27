"""Which paths are reviewable, and the reason recorded for every one that is not.

Files are dropped whole rather than trimmed, and every exclusion is reported in `<excluded_files>`
with a reason. Most decisions come from `git diff --numstat` before any content is fetched:
`DiffExclusionPolicy`, binary files, files over `MAX_DIFF_LINES_PER_FILE`, and files
`.gitattributes` marks as generated. The rest can only be decided from the content -- see `diffs`
-- which is also what `only_instructions` cannot apply, since that mode deliberately fetches
nothing.

Untracked files come into scope in `working_tree` mode only, capped at `MAX_UNTRACKED_FILES` before
anything is asked of git about them. Binary and oversized cannot be settled here for them, because
there is no `--numstat` row to read.

The executor returns the head and tail of any output over about 50KB with the middle cut out, and a
listing read that way would drop files from the review without naming them. `check-attr` is asked
in batches of `CHECK_ATTR_BATCH` paths so its answer stays under that cap, and every listing that
decides scope -- `status`, `--numstat`, `check-attr` -- is refused outright if it was cut anyway.

There is no cap on the assembled total, matching ambient review. A change large enough to pass the
tool's byte-truncation threshold is therefore sliced by `DuoBaseTool._arun`, landing mid-hunk and
dropping later files unnamed. Ambient review does the same, so this is a shared gap; closing it
needs a budget over the formatted blocks.

Known gaps against ambient review
---------------------------------
Untracked files are read as UTF-8 and anything else is excluded as `binary or non-UTF-8 content`.
A latin-1 or UTF-16 source file is therefore dropped rather than reviewed. The reason names what
happened, and the alternative -- guessing an encoding -- risks handing the reviewer mojibake to
comment on. Tracked files are unaffected: their diffs come from git, not from a file read.

Generated-file exclusion reads only `.gitattributes`, whereas ambient review's `generated_file`
comes from Gitaly, which falls back to `enry.IsGenerated`. Files enry infers from path or content
-- lockfiles, minified bundles, vendored trees -- are excluded there and reviewed here. Closing it
needs a Gitaly-backed API for an arbitrary ref pair or a port of enry, neither of which fits a
possibly-unpushed target.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import structlog
from langchain_core.tools import ToolException

from ai.features.review.code_review.components.local_review.git_output import (
    FILTER_ATTRIBUTE,
    GENERATED_ATTRIBUTES,
    FileChange,
    complete_listing,
    parse_check_attr,
)
from ai.features.review.code_review.components.local_review.transport import (
    GitRunner,
    is_denylisted,
)
from duo_workflow_service.policies.diff_exclusion_policy import DiffExclusionPolicy

logger = structlog.stdlib.get_logger(__name__)

MAX_DIFF_LINES_PER_FILE = 5000
MAX_UNTRACKED_FILES = 20
# Three attribute triples per path; at 50 even long paths stay well under the executor's output cap.
CHECK_ATTR_BATCH = 50

_EXCLUSION_BINARY = "binary file"
EXCLUSION_NON_TEXT = "binary or non-UTF-8 content"
EXCLUSION_UNREADABLE = "could not be read from the checkout"
EXCLUSION_OVERSIZED = f"more than {MAX_DIFF_LINES_PER_FILE} changed lines"
_EXCLUSION_POLICY = "excluded by project policy"
_EXCLUSION_DENYLIST = "excluded by Duo context exclusions"
_EXCLUSION_GENERATED = "generated file"
_EXCLUSION_UNTRACKED_CAP = f"more than {MAX_UNTRACKED_FILES} untracked files"
_EXCLUSION_FILTERED = (
    "managed by a git filter, so the worktree content is not what git stores"
)
EXCLUSION_TRUNCATED = "diff too large for the executor to return in one piece"
EXCLUSION_NO_DIFF = "git returned no diff for this path"


async def attribute_paths(
    git: GitRunner,
    paths: List[str],
    source: Optional[str],
    drivers: Tuple[str, ...] = (),
) -> Tuple[Set[str], Set[str]]:
    """Paths git's attributes mark as generated, and those a filter driver rewrites.

    `check-attr` reads attributes only, never content, so it cannot trigger a filter itself.
    """
    generated: Set[str] = set()
    filtered: Set[str] = set()

    for start in range(0, len(paths), CHECK_ATTR_BATCH):
        batch = paths[start : start + CHECK_ATTR_BATCH]
        try:
            raw = await _check_attr(git, source, batch, drivers)
        except ToolException as error:
            if not source or not _rejects_source_option(error):
                raise
            # `--source` needs git >= 2.40; fall back rather than failing the review.
            logger.info(
                "check-attr --source unsupported, using working-tree attributes"
            )
            source = None
            raw = await _check_attr(git, source, batch, drivers)

        batch_generated, batch_filtered = parse_check_attr(
            complete_listing(raw, "check-attr")
        )
        generated |= batch_generated
        filtered |= batch_filtered

    return generated, filtered


async def _check_attr(
    git: GitRunner, source: Optional[str], paths: List[str], drivers: Tuple[str, ...]
) -> str:
    argv = ["check-attr"]
    if source:
        argv.append(f"--source={source}")
    argv += ["-z", *GENERATED_ATTRIBUTES, FILTER_ATTRIBUTE, "--", *paths]

    return await git.run(*argv, filter_drivers=drivers)


def _rejects_source_option(error: ToolException) -> bool:
    # Older git prints `error: unknown option 'source=<sha>'`; any other failure is a real one.
    message = str(error)
    return "unknown option" in message and "source" in message


def select_files(
    project: Any,
    numstat: Dict[str, FileChange],
    generated: Set[str],
    filtered: Set[str],
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Decide every `--numstat` path from policy, denylist, attributes and its numbers, in that order."""
    policy = DiffExclusionPolicy(project)
    included: List[str] = []
    excluded: List[Tuple[str, str]] = []

    for path in sorted(numstat):
        change = numstat[path]
        if not policy.is_allowed(path) or not policy.is_allowed(change.old_path):
            excluded.append((path, _EXCLUSION_POLICY))
        elif is_denylisted(path) or is_denylisted(change.old_path):
            excluded.append((path, _EXCLUSION_DENYLIST))
        elif path in generated:
            excluded.append((path, _EXCLUSION_GENERATED))
        elif path in filtered:
            excluded.append((path, _EXCLUSION_FILTERED))
        elif change.binary:
            excluded.append((path, _EXCLUSION_BINARY))
        elif change.changed_lines > MAX_DIFF_LINES_PER_FILE:
            excluded.append((path, EXCLUSION_OVERSIZED))
        else:
            included.append(path)

    return included, excluded


def select_untracked(
    project: Any, untracked: List[str]
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Path-level selection for untracked files, before git is asked anything about them.

    The cap is applied here rather than after `attribute_paths`, so a checkout with thousands of untracked files never
    puts them all on one `check-attr` command line. Binary and oversized are decided from content, so unlike tracked
    files they cannot be settled here.
    """
    policy = DiffExclusionPolicy(project)
    candidates: List[str] = []
    excluded: List[Tuple[str, str]] = []

    for path in untracked:
        if not policy.is_allowed(path):
            excluded.append((path, _EXCLUSION_POLICY))
        elif is_denylisted(path):
            excluded.append((path, _EXCLUSION_DENYLIST))
        else:
            candidates.append(path)

    for path in candidates[MAX_UNTRACKED_FILES:]:
        excluded.append((path, _EXCLUSION_UNTRACKED_CAP))

    return candidates[:MAX_UNTRACKED_FILES], sorted(excluded)


def exclude_attributed(
    paths: List[str], generated: Set[str], filtered: Set[str]
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Drop the paths whose attributes exclude them, once `attribute_paths` has answered for them."""
    allowed: List[str] = []
    excluded: List[Tuple[str, str]] = []

    for path in paths:
        if path in generated:
            excluded.append((path, _EXCLUSION_GENERATED))
        elif path in filtered:
            excluded.append((path, _EXCLUSION_FILTERED))
        else:
            allowed.append(path)

    return allowed, sorted(excluded)
