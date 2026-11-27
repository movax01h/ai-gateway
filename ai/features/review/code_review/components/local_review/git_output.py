"""Parsers for git's machine-readable output, and the records they produce.

Two coordinate systems have to be reconciled. `git diff --numstat` reports
repository-root-relative paths, and those paths are fed straight back as the pathspec that
fetches the content -- but pathspecs are read relative to the working directory, and
`--literal-pathspecs` rules out the `:(top)` escape. Below the root the two disagree, the
pathspec matches nothing, and the review reports an empty diff having found changes.
`--relative` settles it for `diff`. `git status --porcelain` has no such option and always
reports from the root, so `parse_status` rebases its paths onto the executor's directory and
drops what falls outside it.
"""

from typing import Dict, List, Optional, Set, Tuple

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.bl_security.executor_output import executor_truncated

# The same attributes Gitaly reads before falling back to enry.
GENERATED_ATTRIBUTES = ("gitlab-generated", "linguist-generated")

# Names the driver git would run over the file's content; the command itself lives in config.
FILTER_ATTRIBUTE = "filter"


class FileChange(BaseModel):
    path: str
    old_path: str
    added: int
    deleted: int
    binary: bool

    @property
    def changed_lines(self) -> int:
        return self.added + self.deleted


class WorkingTreeStatus(BaseModel):
    modified: List[str] = Field(default_factory=list)
    untracked: List[str] = Field(default_factory=list)

    @property
    def is_dirty(self) -> bool:
        return bool(self.modified or self.untracked)


def complete_listing(raw: str, listing: str) -> str:
    """Refuse a listing the executor cut, which would otherwise drop files from the review unnamed."""
    if executor_truncated(raw):
        raise ToolException(
            f"Error: the git {listing} output is too large for the executor to return in one"
            " piece, so this change cannot be reviewed in full from the checkout"
        )

    return raw


def parse_check_attr(raw: str) -> Tuple[Set[str], Set[str]]:
    """Read `git check-attr -z` output, which is a flat run of path/attribute/value triples.

    Returns the generated paths and the filter-driven ones.
    """
    fields = raw.split("\0")
    generated: Set[str] = set()
    filtered: Set[str] = set()

    for index in range(0, len(fields) - 2, 3):
        path, attribute, value = fields[index], fields[index + 1], fields[index + 2]
        if attribute == FILTER_ATTRIBUTE:
            # This attribute names a driver, so anything but the two sentinels means filtered.
            if value not in ("unspecified", "unset"):
                filtered.add(path)
        # `unset` deliberately means not generated; `unspecified` means nothing declared.
        elif value == "set":
            generated.add(path)

    return generated, filtered


def parse_status(raw: str, prefix: str = "") -> WorkingTreeStatus:
    """Read `git status --porcelain -z`, rebasing root-relative paths onto `prefix` and dropping those outside it."""
    modified: List[str] = []
    untracked: List[str] = []

    entries = [entry for entry in raw.split("\0") if entry]
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if len(entry) < 4:
            continue

        code, path = entry[:2], entry[3:]
        # Consume the origin field before any `continue`, or later entries shift by one.
        renamed = "R" in code or "C" in code
        if renamed:
            index += 1

        scoped = _strip_prefix(path, prefix)
        if scoped is None:
            # Outside the executor's directory, so nothing in the flow could open it.
            continue

        if code == "??":
            untracked.append(scoped)
        else:
            modified.append(scoped)

    return WorkingTreeStatus(modified=sorted(modified), untracked=sorted(untracked))


def parse_numstat(raw: str) -> Dict[str, FileChange]:
    """Read `git diff --numstat -z` into one `FileChange` per new path, carrying the old path of a rename."""
    changes: Dict[str, FileChange] = {}

    fields = [field for field in raw.split("\0") if field != ""]
    index = 0
    while index < len(fields):
        counts = fields[index].split("\t", 2)
        index += 1
        if len(counts) < 2:
            continue

        added, deleted = counts[0], counts[1]
        inline_path = counts[2] if len(counts) > 2 else ""

        if inline_path:
            old_path = new_path = inline_path
        else:
            # A rename emits an empty third column, then old and new paths as separate fields.
            if index + 1 >= len(fields):
                break
            old_path, new_path = fields[index], fields[index + 1]
            index += 2

        binary = added == "-" or deleted == "-"
        changes[new_path] = FileChange(
            path=new_path,
            old_path=old_path,
            added=0 if binary else int(added),
            deleted=0 if binary else int(deleted),
            binary=binary,
        )

    return changes


def is_text(value: str) -> bool:
    """Whether decoded content is text: no NUL and no replacement character left by a failed decode."""
    return "\0" not in value and "\ufffd" not in value


def _strip_prefix(path: str, prefix: str) -> Optional[str]:
    """Re-express a repository-root-relative path against `prefix`, or None if it is outside."""
    if not prefix:
        return path

    return path[len(prefix) :] if path.startswith(prefix) else None
