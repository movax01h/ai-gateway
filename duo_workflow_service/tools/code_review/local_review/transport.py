"""The executor boundary: every git command and file read this package makes goes through here.

`runCommand` reports a non-zero exit as a *successful* action and prefixes the body with
`Exit code: <n>`, so `_unwrap_command_output` strips the envelope and raises `GitCommandError`
carrying the code and git's diagnostic lines only: git flushes stdout before it dies, so a failed
`diff` can carry hunks that no exclusion has seen, and those must reach neither a log nor the
model. A response over the executor's size limit puts its notice *before* that header; the notice
is stripped to find the header and kept on the body, so `executor_output.executor_truncated` still
reports the cut to whoever parses the output.

`GIT_GLOBAL_FLAGS` is `read_only_git`'s set of pins against a poisoned `.git/config`, one for every
git boundary, plus `--literal-pathspecs`. The pins this package leans on most are
`core.warnAmbiguousRefs=false`, because `rev-parse` and `merge-base` output is read as a bare SHA
and the executor merges stderr into it, and the lazy-fetch refusal, because a partial clone would
otherwise fetch missing blobs through programs named in its own config; that case surfaces as
`PARTIAL_CLONE_MESSAGE` instead.

Configured git filters
----------------------
A `filter` attribute makes git run a configured clean/smudge/process driver to canonicalise
working-tree content. `git status` does this whenever stat alone cannot settle whether a file
changed, and a working-tree `git diff` always does. `--no-ext-diff` and `--no-textconv` do
not cover it -- they disable `diff.external` and `diff.*.textconv`, which are different
mechanisms -- so without the handling below this tool would execute a configured program, and
could write to the filesystem, with no approval prompt. Same threat model as the
`core.fsmonitor` override: the driver command comes from git config, not from the in-tree
`.gitattributes`.

Git has no global switch for this, so `filter_drivers` enumerates the configured drivers and `run`
empties each one by name, before the first command that reads the working tree. A name containing
`=` cannot be expressed with `-c` at all, and `run` refuses rather than leave that driver live; a
listing the executor truncated is refused for the same reason.
Tree-to-tree diffs never convert worktree content and are unaffected either way.

That enumeration reads the superproject's config only, while `status` and a working-tree `diff`
recurse into submodules, whose own config may name drivers of its own. `--ignore-submodules=all`
in `DIFF_FLAGS` and `STATUS_FLAGS` stops the recursion, and `run` refuses either command without
it. It has to be on the command line:
`submodule.<name>.ignore` in `.gitmodules`, which is part of the tree under review, overrides
`diff.ignoreSubmodules`.

Emptying a driver means git hashes the raw bytes instead, so a file the driver would have
rewritten -- a Git LFS object is the common case -- would diff as pointer against content. Those
paths are excluded outright, by `exclusions.attribute_paths`, rather than reported as a diff that
never existed.

Running below the repository root
---------------------------------
The executor runs git in the workspace folder, which need not be the repository root. Everything
this package reports is relative to that folder, and the review is scoped to it. Scoping rather
than reporting the whole repository is deliberate: the executor refuses any read resolving outside
the workspace folder, so a changed file above it can be named but never opened, and listing one in
`<changed_files>` would hand the reviewer a coverage contract it cannot discharge.

`--relative` in `DIFF_FLAGS` settles the coordinates for `diff`. `git status --porcelain` has no
such option and always reports from the root, so `git_output.parse_status` rebases its paths with
`prefix`.
"""

from typing import Any, Dict, Optional, Set, Tuple

import structlog
from langchain_core.tools import ToolException

from contract import contract_pb2
from duo_workflow_service.bl_security.executor_output import (
    EXIT_CODE_HEADER,
    executor_truncated,
    strip_size_limit_notice,
)
from duo_workflow_service.executor.action import _execute_action, _read_file_fully
from duo_workflow_service.tools.code_review.custom_instructions import (
    INSTRUCTIONS_FILE_PATH,
)
from duo_workflow_service.tools.filesystem import validate_duo_context_exclusions
from duo_workflow_service.tools.read_only_git import (
    GIT_GLOBAL_FLAGS as HARDENED_GIT_GLOBAL_FLAGS,
)
from duo_workflow_service.tools.read_only_git import (
    PARTIAL_CLONE_MESSAGE,
    REFUSED_LAZY_FETCH,
    diagnostics,
)

logger = structlog.stdlib.get_logger(__name__)

# Paths here are relative to the executor's directory and go after `--` as data, where a file
# named `*` must not match every file. `read_only_git` uses `:(top,literal)` on each pathspec
# instead, which anchors at the repository root and would undo `--relative`.
GIT_GLOBAL_FLAGS = (*HARDENED_GIT_GLOBAL_FLAGS, "--literal-pathspecs")

# Asserted on the way out, so no later edit can make this tool write to the repo.
ALLOWED_GIT_SUBCOMMANDS = frozenset(
    {
        "check-attr",
        "config",
        "diff",
        "merge-base",
        "rev-parse",
        "status",
    }
)

# The allowlist matches on subcommand alone, and `git config` can also write. `-z` ends each entry
# with NUL and puts a newline between key and value, so a driver name containing a space survives.
_CONFIG_READ_ARGV = ("config", "-z", "--get-regexp", r"^filter\.")

# `--no-ext-diff` and `--no-textconv` are positional, so a later flag re-enables the program,
# `--output` makes diff write a file, and `--no-index` diffs paths outside the repository.
# Checked on every argument, as `read_only_git` does: whether `--` ends the options is a per-subcommand
# fact (`log` pre-parses with parse-options, where an option can take `--` as its value).
_DENIED_OPTIONS = ("--output", "--ext-diff", "--textconv", "--no-index")

# `required` is set to false rather than emptied: git reads an emptied command as a filter
# that failed, not one that is absent, and a required filter failing aborts the command.
_FILTER_OVERRIDES = (
    ("clean", ""),
    ("smudge", ""),
    ("process", ""),
    ("required", "false"),
)

_FILTER_KEYS = tuple(key for key, _ in _FILTER_OVERRIDES)

# External diff/textconv run configured programs; `--relative` sets the coordinates.
DIFF_FLAGS = (
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "-M",
    "--relative",
    "--ignore-submodules=all",
)

STATUS_FLAGS = ("--porcelain", "-z", "-uall", "--ignore-submodules=all")

# `run` refuses a `diff` or `status` without these, so the no-program guarantee is not a call-site
# choice either. The flags that would undo them later in argv are denied.
_REQUIRED_FLAGS = {
    "diff": ("--no-ext-diff", "--no-textconv", "--ignore-submodules=all"),
    "status": ("--ignore-submodules=all",),
}

# Git takes the last spelling: `status` accepts the negation and both take `=none`.
_SUBMODULE_OPTIONS = ("--ignore-submodules", "--no-ignore-submodules")

# Literal paths only, never model-controlled: the instructions file lives under denylisted `.gitlab/duo`.
_DENYLIST_EXEMPT_PATHS = frozenset({INSTRUCTIONS_FILE_PATH})


class GitCommandError(ToolException):
    """A git command that ran and exited non-zero, carrying the code it exited with."""

    def __init__(self, subcommand: str, code: str, body: str) -> None:
        super().__init__(
            f"Error: git {subcommand} exited with {code}: {diagnostics(body)}"
        )
        self.code = code


class GitRunner:
    """Runs read-only git commands and file reads on the executor."""

    def __init__(self, metadata: Dict[str, Any]) -> None:
        self.metadata = metadata

    async def run(self, *argv: str, filter_drivers: Tuple[str, ...] = ()) -> str:
        """Run one read-only git command on the executor and return its output.

        Args:
            argv: The git subcommand and its arguments, without the `git` program. The subcommand must be in
                `ALLOWED_GIT_SUBCOMMANDS`, options that write a file or run a configured program are refused,
                and `git config` is accepted only as the filter query this package issues.
            filter_drivers: Names of configured filter drivers, as returned by `filter_drivers`, each emptied
                with `-c` before the subcommand. Only the name comes from the caller, so no override can land
                outside `filter.<name>.*` or undo the pins in `GIT_GLOBAL_FLAGS`.

        Returns:
            The command's output with the executor's exit-code header removed.

        Raises:
            ToolException: The argv is empty, not read-only or missing a safety flag, or a filter driver cannot
                be disabled with `-c`.
                Nothing is sent to the executor.
            GitCommandError: The command ran and exited non-zero.
        """
        if not argv:
            raise ToolException("Error: git was given no subcommand")
        if argv[0] not in ALLOWED_GIT_SUBCOMMANDS:
            raise ToolException(f"Error: git {argv[0]} is not a read-only command")
        denied = _denied_option(argv)
        if denied:
            raise ToolException(f"Error: git option {denied} is not allowed")
        missing = _missing_flag(argv)
        if missing:
            raise ToolException(f"Error: git {argv[0]} must be run with {missing}")

        if argv[0] == "config" and tuple(argv) != _CONFIG_READ_ARGV:
            raise ToolException(
                "Error: git config is allowed only as a read-only query"
            )

        raw = await _execute_action(
            self.metadata,
            contract_pb2.Action(
                runCommand=contract_pb2.RunCommandAction(
                    program="git",
                    arguments=[
                        *GIT_GLOBAL_FLAGS,
                        *_filter_overrides(filter_drivers),
                        *argv,
                    ],
                    flags=[],
                )
            ),
        )

        return _unwrap_command_output(raw, argv[0])

    async def read_file(self, file_path: str) -> Optional[str]:
        """Read a file from the executor's workspace in full.

        Args:
            file_path: The path, relative to the executor's directory.

        Returns:
            The whole file, or None when the path is on the managed Duo context-exclusion denylist, outside
            `_DENYLIST_EXEMPT_PATHS`, or the read fails for any reason. A denylisted path sends no action to
            the executor.
        """
        if file_path not in _DENYLIST_EXEMPT_PATHS and is_denylisted(file_path):
            return None

        try:
            # The executor pages reads and reports the first page as success; this follows
            # every page or raises, so a prefix is never returned as the file.
            return await _read_file_fully(
                self.metadata, file_path, execute=_execute_action
            )
        except Exception:
            logger.info(
                "Local review context could not read a file, continuing without it",
                file_path=file_path,
            )
            return None

    async def prefix(self) -> str:
        """The executor's directory relative to the repository root, empty when it is the root.

        Everything this tool reports is relative to the executor's directory, because that is what
        the reviewer's own file tools can open: the executor refuses any read that resolves outside
        the workspace folder. `--relative` puts `diff` in those terms; this is for `status`, which
        has no equivalent.
        """
        return (await self.run("rev-parse", "--show-prefix")).strip()

    async def filter_drivers(self) -> Tuple[str, ...]:
        """Names of the configured clean/smudge/process drivers, for `run(filter_drivers=...)` to empty.

        `--no-ext-diff` and `--no-textconv` do not cover these. A `filter` attribute makes git run
        the configured driver to canonicalise working-tree content, which both `git status` and a
        working-tree `git diff` do -- arbitrary execution from a tool that never asks for approval.
        There is no global switch, so `run` empties every configured driver individually by name,
        and clears its `required` flag so the emptied command is not read as a failure.
        """
        try:
            raw = await self.run(*_CONFIG_READ_ARGV)
        except GitCommandError as error:
            # Only exit 1 means no filter is configured. Anything else -- a corrupt config
            # exits 128 -- must not leave every driver live by returning no drivers.
            if error.code != "1":
                raise

            return ()

        if executor_truncated(raw):
            # The executor keeps the head and tail of a long listing; a driver in the cut would stay live.
            raise ToolException(
                "Error: the git filter configuration is too large to read in full, so the"
                " working tree cannot be read safely"
            )

        return tuple(sorted(_filter_driver_names(raw)))


def _filter_overrides(drivers: Tuple[str, ...]) -> Tuple[str, ...]:
    for name in drivers:
        if "=" in name:
            # `-c` splits at the first `=`: `filter.a=b.clean=` sets `filter.a` and leaves `a=b` live.
            raise ToolException(
                f"Error: git filter driver {name!r} cannot be disabled with -c"
            )

    return tuple(
        flag
        for name in drivers
        for key, value in _FILTER_OVERRIDES
        for flag in ("-c", f"filter.{name}.{key}={value}")
    )


def _denied_option(argv: Tuple[str, ...]) -> Optional[str]:
    return next((arg for arg in argv if _is_denied(arg)), None)


def _is_denied(arg: str) -> bool:
    if arg.startswith(_DENIED_OPTIONS):
        return True

    return arg.startswith(_SUBMODULE_OPTIONS) and arg != "--ignore-submodules=all"


def _missing_flag(argv: Tuple[str, ...]) -> Optional[str]:
    return next(
        (flag for flag in _REQUIRED_FLAGS.get(argv[0], ()) if flag not in argv), None
    )


def is_denylisted(path: str) -> bool:
    """Whether the managed Duo context-exclusion denylist blocks this path."""
    if not path:
        return False

    try:
        validate_duo_context_exclusions(path)
    except ToolException:
        return True

    return False


def _unwrap_command_output(raw: str, subcommand: str) -> str:
    """Strip the executor's exit-code envelope, raising when the command actually failed.

    Without this every parser here reads `Exit code: 0` as its first line of data, and a failed command is
    indistinguishable from a successful one because the action still reports success. A response over the executor's
    size limit puts its notice *before* the header; it is stripped to find the header and kept on the returned body so
    `executor_truncated` still sees it. An executor that returns the output bare is tolerated, so the tool works either
    way.
    """
    text = strip_size_limit_notice(raw)
    notice = raw[: len(raw) - len(text)]

    match = EXIT_CODE_HEADER.match(text)
    if not match:
        return raw

    body = text[match.end() :]
    # Compared as text: the header admits any non-space token, so a non-numeric code is a
    # failure to report rather than a parse error to raise from here.
    code = match.group(1)
    if code != "0":
        if REFUSED_LAZY_FETCH.search(body):
            raise ToolException(PARTIAL_CLONE_MESSAGE.format(subcommand=subcommand))
        raise GitCommandError(subcommand, code, body)

    return notice + body


def _filter_driver_names(raw: str) -> Set[str]:
    """Driver names from `git config -z --get-regexp ^filter\\.`; a name may itself contain dots or spaces."""
    names = set()

    for entry in raw.split("\0"):
        key = entry.split("\n", 1)[0]
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] == "filter" and parts[-1] in _FILTER_KEYS:
            names.add(".".join(parts[1:-1]))

    return names
