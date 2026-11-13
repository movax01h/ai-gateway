"""Local read-only git history: the working-tree counterpart of ``list_commits``.

The API tool this complements reads the merge request's project over REST, so it cannot
see a local, possibly-unpushed change. This reads the checkout instead, through the
hardened egress point in ``read_only_git``. Both stay registered, under different
privileges, so the tool description names the other and says when to use which.

Output is never rendered in the ``<file_diff>``/``<line>`` structure the review target
uses. That structure is an anchoring contract: the reviewer prompt instructs the model
to copy ``filename`` and read ``new_line`` verbatim from ``<line>`` tags, so rendering
history the same way would tell it that historical lines are anchorable. A finding
anchored outside the review target's hunks is silently withheld when published, which
is the worst outcome for a flow whose design rests on never losing a candidate.

Shallow clones are the normal case, not an edge case -- GitLab CI's default ``GIT_DEPTH``
truncates history in ambient executors -- and truncated history reads exactly like an
absent commit. Every answer that could be changed by truncation says so.
"""

import re
from typing import Any, ClassVar, List, Optional, Tuple, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.bl_security.executor_output import (
    executor_truncated,
    strip_size_limit_notice,
)
from duo_workflow_service.policies.diff_exclusion_policy import DiffExclusionPolicy
from duo_workflow_service.tools.read_only_git import (
    DIFF_FLAGS,
    SHA_RE,
    ReadOnlyGitTool,
    exclusion_reason,
    literal_pathspec,
    validate_repo_path,
)

__all__ = [
    "ListLocalCommits",
]

_MAX_COMMITS = 100


_MAX_FILES_PER_COMMIT = 20


# NUL between fields and nothing else: no subject, author name or path can contain one, so
# nothing in a commit can forge a record boundary. Any other separator byte can.
_LOG_FORMAT = "--format=%h%x00%ad%x00%an%x00%P%x00%s"
_LOG_FIELDS = 5

_STATUS_RE = re.compile(r"^[A-Z]\d*\Z")

# The Node executor (gitlab-lsp, workflow_executor/.../command_output_truncation.ts)
# splices only output over 50 KB and fills that budget to within a few bytes, so a marker
# in a smaller body was written by git: a path or subject can carry it. A smaller splice
# still fails closed, because it misaligns the record stream and `_collect` notices.
_SPLICE_FLOOR_BYTES = 49 * 1024

# One `--name-status` entry: the status letter and its one path, or two for a rename/copy.
Change = Tuple[str, List[str]]
# One commit: its format fields, then its changes.
Record = Tuple[List[str], List[Change]]


# Without this `git log --name-status` lists no files at all for a merge. It implies -p,
# so it must be paired with --name-status or it dumps every merge's full patch. It needs
# git 2.31 or later, the floor `ReadOnlyGitTool` enforces: `-m` would list a merge's
# files against every parent, and `--first-parent` would change which commits are walked.
_MERGE_FLAG = "--diff-merges=first-parent"


_HISTORY_IS_EVIDENCE = (
    "These commits are history, not the change under review. Use them as evidence for "
    "whether removed or rewritten behaviour was load-bearing."
)

_WAS_TRUNCATED = (
    "The executor cut the middle out of this output, so it is incomplete. Treat what is "
    "missing as unknown rather than absent, and narrow the request to see it in full."
)

_SHALLOW_BANNER = (
    "<history_may_be_incomplete>\n"
    "This is a shallow clone, so history stops at a graft boundary. A commit missing "
    "from this result may still exist; absence here is not evidence.\n"
    "</history_may_be_incomplete>"
)
# At a graft git has no parent to diff against and lists the whole tree as added.
_GRAFT_NOTE = (
    "(at the shallow clone's boundary: its parents are not available, so its file list "
    "would be the whole tree, not what it changed)"
)

# Our own delimiters, neutralised in every span of contributor-authored text before it is
# embedded. `encode_dangerous_tags` only covers SECURITY_DELIMITER_TAGS, and adding these
# to that set would encode the envelope this module writes.
_ENVELOPE_TAG_RE = re.compile(
    r"<(/?(?:git_history|history_may_be_incomplete|file_diff|line)\b)"
)


def _neutralise(text: str) -> str:
    return _ENVELOPE_TAG_RE.sub(r"&lt;\1", text)


_PATH_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _printable(path: str) -> str:
    # `-z` hands paths over raw, and a newline in one would start a forged entry.
    return _PATH_CONTROL_RE.sub(lambda m: f"\\x{ord(m.group()):02x}", path)


def _attr(value: str) -> str:
    return _printable(_neutralise(value)).replace('"', "&quot;")


class _Commit(BaseModel):
    short_sha: str
    date: str
    author: str
    subject: str
    changed_files: List[str] = Field(default_factory=list)

    def render(self) -> str:
        head = f"- {self.short_sha} {self.date} {_neutralise(self.author)}: {_neutralise(self.subject)}"
        if not self.changed_files:
            return head
        files = "\n".join(f"  {_neutralise(entry)}" for entry in self.changed_files)
        return f"{head}\n{files}"


def _parse_log(raw: str, arity: int) -> Tuple[List[Record], bool]:
    """Parse ``git log -z`` output whose format is *arity* NUL-separated fields.

    git emits ``<fields>\\0`` per commit and, when the commit has a diff, a block of
    ``--name-status`` tokens whose first carries a leading newline:
    ``\\n<status>\\0<path>\\0...``. The next commit's first field follows the last path
    directly, so the stream is read by arity: *arity* fields, then entries for as long as
    the next token is a status code. The flag says whether every token was consumed;
    git writes whole records, so tokens left over mean the stream was shifted.
    """
    tokens = raw.split("\0")
    if tokens and tokens[-1] == "":
        tokens.pop()

    records: List[Record] = []
    index = 0
    while index + arity <= len(tokens):
        fields = tokens[index : index + arity]
        index += arity
        changes, index = _parse_changes(tokens, index)
        records.append((fields, changes))

    return records, index == len(tokens)


def _parse_changes(tokens: List[str], index: int) -> Tuple[List[Change], int]:
    changes: List[Change] = []
    if index < len(tokens) and tokens[index].startswith("\n"):
        tokens[index] = tokens[index][1:]

    while index < len(tokens) and _STATUS_RE.match(tokens[index]):
        status = tokens[index]
        # A rename or copy carries both paths, so it consumes one token more.
        wants = 2 if status[0] in "RC" else 1
        paths = tokens[index + 1 : index + 1 + wants]
        index += 1 + wants
        # A truncated block can end on a status with its path cut off.
        if len(paths) == wants:
            changes.append((status, paths))

    return changes, index


def _executor_cut(raw: str) -> bool:
    if strip_size_limit_notice(raw) != raw:
        return True
    return len(raw.encode()) >= _SPLICE_FLOOR_BYTES and executor_truncated(raw)


def _spans_splice(fields: List[str], changes: List[Change], truncated: bool) -> bool:
    """Whether this record is the one the executor's cut landed in.

    git never writes a newline into a log field, so one there is the cut whatever the output's size. A path can carry
    one, so the paths count only once the size gate has called the output cut.
    """
    if any("\n" in field for field in fields):
        return True
    paths = [path for _, entry in changes for path in entry]
    return truncated and executor_truncated("\0".join(paths))


def _changed_files(changes: List[Change], policy: DiffExclusionPolicy) -> List[str]:
    """Render a commit's file list, marking paths the policy would not let the model read.

    Names are shown with the reason, as the review context does for the user; only content
    is withheld. Only the requested ``path`` is validated up front; the names git returns
    are not, and a rename exposes its old name too, so every side of every entry is checked.
    """
    entries: List[str] = []
    for status, paths in changes:
        entry = f"{status} {' -> '.join(map(_printable, paths))}"
        reason = next(filter(None, (exclusion_reason(p, policy) for p in paths)), None)
        if reason:
            entry += f" ({reason})"
        entries.append(entry)

    return _cap(entries, _MAX_FILES_PER_COMMIT)


def _collect(
    records: List[Record],
    policy: DiffExclusionPolicy,
    include_changed_files: bool,
    truncated: bool,
    shallow: bool,
) -> Tuple[List[_Commit], bool]:
    """Turn parsed records into commits, keeping only those before a splice.

    Only a cut puts a newline in a field or a non-SHA where a record starts, and it misaligns every later token. One the
    size gate did not see is still a cut: the records are read again knowing that, so the notice is rendered. Nothing
    after the straddling record is kept: a shifted record opens on whatever field the shift put first, and a contributor
    can make a subject look like a SHA. Nothing is dropped from an output the executor did not cut.
    """
    commits: List[_Commit] = []
    for fields, changes in records:
        short_sha, date, author, parents, subject = fields
        if _spans_splice(fields, changes, truncated) or not SHA_RE.match(short_sha):
            if truncated:
                break
            return _collect(records, policy, include_changed_files, True, shallow)
        changed_files: List[str] = []
        if include_changed_files:
            if shallow and not parents.split():
                changed_files = [_GRAFT_NOTE]
            else:
                changed_files = _changed_files(changes, policy)
        commits.append(
            _Commit(
                short_sha=short_sha,
                date=date,
                author=author,
                subject=subject,
                changed_files=changed_files,
            )
        )
    return commits, truncated


def _cap(entries: List[str], limit: int) -> List[str]:
    if len(entries) <= limit:
        return entries
    return [*entries[:limit], f"... and {len(entries) - limit} more"]


class ListLocalCommitsInput(BaseModel):
    path: Optional[str] = Field(
        default=None,
        description=(
            "Repository-relative file or directory to scope the history to, spelled as "
            "it appears in the diff. Omit for the whole repository. A single path only."
        ),
    )
    ref: Optional[str] = Field(
        default=None,
        description=(
            "Branch, tag or commit SHA to walk back from. Defaults to the checked-out "
            "HEAD. Revision expressions such as 'HEAD~3' are not accepted."
        ),
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=_MAX_COMMITS,
        description="Maximum number of commits to return.",
    )
    include_changed_files: bool = Field(
        default=False,
        description=(
            "Also list the files each commit touched. When `path` is given, only "
            "changes under that path are listed, not the whole commit."
        ),
    )


class ListLocalCommits(ReadOnlyGitTool):
    name: str = "list_local_commits"
    description: str = (
        "List commits from the local git checkout, newest first, for judging whether "
        "code the change removed or rewrote was load-bearing.\n"
        "Scope it to the file you are investigating with `path`.\n"
        "Reads the checkout on disk, so it sees local commits that were never pushed. "
        "Prefer this over list_commits when reviewing a local checkout; use "
        "list_commits for a project that is not checked out.\n"
        "Runs only read-only git commands and never writes to the repository.\n"
        "Follows renames when `path` names a single file, so the history survives a "
        "file move. A moved directory's earlier history is not followed.\n"
        "Examples:\n"
        "- list_local_commits(path='app/models/user.rb')\n"
        "- list_local_commits(path='app/models/user.rb', include_changed_files=True)"
    )
    args_schema: Type[BaseModel] = ListLocalCommitsInput
    allowed_git_subcommands: ClassVar[frozenset[str]] = frozenset({"rev-parse", "log"})

    async def _execute(self, **kwargs: Any) -> str:
        path = kwargs.get("path")
        try:
            limit = int(kwargs.get("limit") or 20)
        except (TypeError, ValueError) as exc:
            raise ToolException(
                f"Error: limit must be a whole number, not {kwargs.get('limit')!r}"
            ) from exc
        limit = min(max(limit, 1), _MAX_COMMITS)
        include_changed_files = bool(kwargs.get("include_changed_files"))

        if path is not None:
            path = validate_repo_path(path, self.project)

        resolved, shallow_clone = await self._resolve_commit(kwargs.get("ref"))

        argv: List[str] = ["log", *DIFF_FLAGS, "-z", "--date=short", "--abbrev=12"]
        argv.append(_LOG_FORMAT)
        argv.append(f"-n{limit}")
        if path is not None:
            # --follow takes exactly one pathspec, which is why `path` is not a list.
            argv.append("--follow")
        if include_changed_files:
            # Paired deliberately: without _MERGE_FLAG a merge lists no files at all,
            # and without --name-status the same flag dumps every merge's full patch.
            argv.extend(["--name-status", _MERGE_FLAG])
        argv.append(resolved)
        argv.append("--")
        if path is not None:
            argv.append(literal_pathspec(path))

        raw = await self._git(*argv)
        # A spliced record loses its leading SHA, so the guard below drops it and the
        # count comes out plausible but short. Reachable with a large `limit` and file
        # lists on a repository whose commits touch many files.
        truncated = _executor_cut(raw)
        # The size-limit notice is left on the body for the check above; parsed as-is it
        # would be glued onto the first record's SHA and drop the newest commit.
        raw = strip_size_limit_notice(raw)

        records, whole = _parse_log(raw, _LOG_FIELDS)
        # The executor merges stderr, so a warning printed before the first record would
        # otherwise read as an empty history.
        if raw.strip() and not (records and SHA_RE.match(records[0][0][0])):
            head = raw.strip().splitlines()[0][:120]
            raise ToolException(
                f"Error: git log output could not be parsed; it began with {head!r}"
            )
        # A cut the size gate did not see still shifts the stream; what it leaves over at
        # the end never starts a record, so only the marker says why.
        truncated = truncated or (not whole and executor_truncated(raw))

        commits, truncated = _collect(
            records,
            DiffExclusionPolicy(self.project),
            include_changed_files,
            truncated,
            shallow_clone,
        )

        # A full page proves nothing: at a graft git goes on filling the page from other
        # branches, so commits behind it that are newer than the last one listed go
        # missing without a shortfall, and with a pathspec the graft itself may be unlisted.
        return self._render(
            commits, path, resolved, include_changed_files, shallow_clone, truncated
        )

    def _render(
        self,
        commits: List[_Commit],
        path: Optional[str],
        resolved: str,
        include_changed_files: bool,
        shallow: bool,
        truncated: bool = False,
    ) -> str:
        notices = []
        if shallow:
            notices.append(_SHALLOW_BANNER)
        if truncated:
            notices.append(_WAS_TRUNCATED)

        if not commits:
            return self._render_empty(path, notices)

        attrs = [f'ref="{resolved[:12]}"', f'count="{len(commits)}"']
        if path is not None:
            attrs.insert(1, f'path="{_attr(path)}"')

        body = "\n".join(commit.render() for commit in commits)
        rendered = (
            f"<git_history {' '.join(attrs)}>\n{body}\n{_HISTORY_IS_EVIDENCE}\n"
            "</git_history>"
        )

        if not include_changed_files:
            scope = " under this path" if path is not None else ""
            rendered += (
                "\nCall again with include_changed_files=True to see which files each "
                f"commit touched{scope}."
            )

        return "\n".join([*notices, rendered])

    @staticmethod
    def _render_empty(path: Optional[str], notices: List[str]) -> str:
        scope = f" touching {_neutralise(path)!r}" if path else ""
        rendered = (
            f'<git_history count="0">\nNo commits{scope}. If a path was '
            "given, check that it is repository-relative and spelled as it appears in "
            "the diff.\n</git_history>"
        )

        return "\n".join([*notices, rendered])

    def format_display_message(
        self, args: ListLocalCommitsInput, _tool_response: Any = None
    ) -> str:
        target = args.path or "the repository"
        return f"List local commit history for {target}"
