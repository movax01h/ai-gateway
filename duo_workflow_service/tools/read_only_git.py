"""One hardened egress point for read-only git commands run on the user's checkout.

Every git action a tool dispatches passes ``ReadOnlyGitTool._git``, which asserts the
subcommand against the tool's own allowlist before anything reaches the executor. Each
allowlist must be a subset of ``READ_ONLY_GIT_SUBCOMMANDS``, checked at class creation
(anything outside it raises ``TypeError``), so least privilege survives: a bug in a
history tool cannot reach ``check-attr``, and no tool can reach a subcommand that writes.

These tools execute ``git`` on the user's machine under the ``read_only_files``
privilege, which is normally pre-approved, so they run without a per-call prompt. The
flags below are what make that safe, and each one blocks a specific way a checkout
arriving with a poisoned ``.git/config`` could turn a read into arbitrary code
execution. Do not drop one without replacing the guarantee.

User-supplied text reaches git through exactly two shapes: paths, which are
prefix-checked, denylist-checked and policy-checked, then passed after ``--`` as a
root-anchored literal pathspec by every call site (asserted per tool, since ``_git``
cannot tell a path from a SHA); and refs, which ``_resolve_commit`` turns into a 40-hex
SHA before any other command sees them. Nothing downstream of that is attacker-shaped.

Git finds its repository by walking up from the executor's directory, so a workspace
that is not itself a repository would read whichever one encloses it, typically a
dotfiles repository in the home directory. ``_resolve_commit`` refuses unless that
directory is the root, or a tracked directory, of the repository git discovered.

Lazy fetching is the one way these commands reach the network. In a partial clone, a
command that needs a missing blob (``-S``, ``-M``, ``--follow``, any patch) fetches it
from the promisor remote, and that fetch runs ``remote.<name>.uploadpack``,
``core.sshCommand``, ``ext::`` commands, credential helpers and, for a protocol name the
repo allows itself, a repo-defined ``alias.remote-<name>`` shell command, all from the
very config this module distrusts. No ``-c`` setting disables the fetch itself (only
``GIT_NO_LAZY_FETCH=1`` does, and ``RunCommandAction`` carries no environment), but the
fetch is a child ``git fetch`` that inherits these pins, and ``fetch.parallel=-1`` makes
it die while reading its config, before any transport or helper is chosen. The child's
``fetch.parallel cannot be negative`` always reaches the output; the command then either
exits 128 with ``could not fetch ... from promisor remote`` or, if the child was already
gone when the command handed it the object ids, dies of SIGPIPE (seen on a loaded CI
runner). The per-protocol pins stay as a second layer, because a repo-level
``protocol.<name>.allow=always`` beats the ``protocol.allow`` default.

That guard exists only from git 2.29, when the lazy fetch became a child process; before
that it ran inside the command and inherited nothing. So the boundary probes ``git
version`` once per tool instance and refuses anything below ``GIT_FLOOR`` before a single
command is sent. The floor is 2.31 rather than 2.29 because ``--diff-merges``, which the
history tools pair with ``--name-status``, needs it anyway.
"""

import re
from typing import Any, ClassVar, NamedTuple, Optional

from langchain_core.tools import ToolException
from pydantic import PrivateAttr

from duo_workflow_service.bl_security.executor_output import (
    EXIT_CODE_HEADER,
    strip_size_limit_notice,
)
from duo_workflow_service.executor.action import _execute_action
from duo_workflow_service.policies.diff_exclusion_policy import DiffExclusionPolicy
from duo_workflow_service.tools.command import (
    resolve_command_timeout,
    run_command_action,
)
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.filesystem import validate_duo_context_exclusions

# The vetted superset. A subcommand only belongs here if it cannot write to the
# repository, the index, the filesystem or the network under any argument this module
# can produce. `test_read_only_git.py` asserts the complement.
# `blame`, `status` and `diff` are deliberately absent. All three are read-only, but they
# convert working-tree content through configured `filter.*` clean/process drivers, which
# is arbitrary execution from a tool that never prompts (`status` and the worktree forms of
# `diff` run the clean driver on every stat-dirty file; verified against real git).
# Everything listed here is tree-to-tree or metadata-only and never runs a driver. Adding a
# working-tree reader means restoring the driver neutralisation the local review transport
# in !6977 carries; do not just add the name.
READ_ONLY_GIT_SUBCOMMANDS = frozenset(
    {
        "check-attr",
        "for-each-ref",
        "log",
        "merge-base",
        "rev-parse",
        "show",
    }
)

# The lazy-fetch guard below holds from 2.29; `--diff-merges=first-parent` needs 2.31.
GIT_FLOOR = (2, 31)
_GIT_VERSION_RE = re.compile(r"\Agit version (\d+)\.(\d+)")

# Never rewrite the index opportunistically; pin the output format against user config
# the parsers cannot read.
GIT_GLOBAL_FLAGS = (
    # Needs git 2.15; the version probe runs with these flags too, so a git too old for
    # any of them fails that probe and is refused by the same path.
    "--no-optional-locks",
    # Only pages onto a tty, but the executor decides that, not this tool.
    "--no-pager",
    "-c",
    "diff.noprefix=false",
    "-c",
    "diff.mnemonicPrefix=false",
    "-c",
    "diff.suppressBlankEmpty=false",
    # git >= 2.45; these override the a/ b/ prefixes diff parsers strip, and neither
    # `noprefix` nor `mnemonicPrefix` constrains them.
    "-c",
    "diff.srcPrefix=a/",
    "-c",
    "diff.dstPrefix=b/",
    # `git status` runs this as a program, so a checkout arriving with a poisoned
    # .git/config would get code execution from a tool that never asks for approval.
    "-c",
    "core.fsmonitor=false",
    # A repo-level diff.submodule=diff spawns `git diff` inside the submodule, out of reach of --no-ext-diff.
    "-c",
    "diff.submodule=short",
    # Runs gpg, and prepends its output to the commit body the parsers split on.
    "-c",
    "log.showSignature=false",
    # Only stops git verifying on its own; `%G?` and `--show-signature` still run these programs.
    "-c",
    "gpg.program=false",
    "-c",
    "gpg.ssh.program=false",
    "-c",
    "gpg.x509.program=false",
    # `--follow` is decided per call; a repo-level value would apply it silently.
    "-c",
    "log.follow=false",
    # A branch and a tag sharing a name make every command print a warning on stderr,
    # which the executor merges into the output `_resolve_commit` expects to be bare hex.
    "-c",
    "core.warnAmbiguousRefs=false",
    # The executor need not run git at the repository root; with this set, `--name-status`
    # paths come out relative to the cwd and no longer match the exclusion policy.
    "-c",
    "diff.relative=false",
    # `-z` keeps `--name-status` paths raw, but the `diff --git` header still octal-escapes
    # non-ASCII names unless this is off, and the patch post-check compares them verbatim.
    "-c",
    "core.quotePath=false",
    # The lazy fetch is a child `git fetch`, which dies on this while reading config.
    "-c",
    "fetch.parallel=-1",
    # Second layer against the same fetch; see the module docstring.
    "-c",
    "protocol.allow=never",
    "-c",
    "protocol.file.allow=never",
    "-c",
    "protocol.git.allow=never",
    "-c",
    "protocol.ssh.allow=never",
    "-c",
    "protocol.http.allow=never",
    "-c",
    "protocol.https.allow=never",
    "-c",
    "protocol.ftp.allow=never",
    "-c",
    "protocol.ftps.allow=never",
    "-c",
    "protocol.ext.allow=never",
    "-c",
    "protocol.fd.allow=never",
)

# `diff.<driver>.textconv` runs a configured program on tree-to-tree `log -p` and `show -p`
# too (`diff.external` only on `git diff`), so these are not a call-site choice: `_git`
# splices them in after the subcommand for every command that can emit a diff. Only there,
# because `rev-parse` echoes options it does not recognise back onto stdout. Git takes the
# last spelling of a flag, so `_git` also refuses the options that would turn them back on.
DIFF_SAFETY_FLAGS = ("--no-ext-diff", "--no-textconv")
_DIFF_SUBCOMMANDS = frozenset({"log", "show"})
# `--ext-diff` and `--textconv` undo the flags above; `--submodule=diff` undoes the
# `diff.submodule` pin and spawns a `git diff` inside the submodule that the flags never
# reach; `--output` makes a read-only tool write a file the caller names. Git rejects
# abbreviated long options, so exact spellings are enough. Checked on every argument, not
# only those before `--`: an option that takes a value, such as `--decorate-refs`, swallows
# a following `--` as that value and git goes on reading options after it.
_GUARANTEE_UNDOING_OPTIONS = frozenset(
    {"--ext-diff", "--textconv", "--output", "--submodule"}
)

DIFF_FLAGS = ("--no-color", "-M")

# `\Z`, not `$`: `$` also matches before a trailing newline, which would let `main\n` through.
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*\Z")
# 64 for SHA-256 repositories, which GitLab supports.
SHA_RE = re.compile(r"^[0-9a-f]{7,64}\Z")
_FULL_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def validate_ref(ref: str) -> str:
    """Return *ref* unchanged, or raise if it could be read as anything but a ref name.

    Rejects ``HEAD~3`` and ``HEAD^`` along with the option-shaped input this exists to
    stop. That is deliberate: an agent investigating history already holds the SHAs from
    its previous call, so revision arithmetic buys nothing and every extra metacharacter
    is ref-parsing surface.
    """
    if not _REF_RE.match(ref) or ".." in ref:
        raise ToolException(
            f"Error: {ref!r} is not a valid git ref name. Use a branch, tag or commit "
            "SHA; revision expressions such as 'HEAD~3' are not accepted."
        )
    return ref


def literal_pathspec(path: str) -> str:
    """Wrap a validated repository-relative path so git reads it as data, from the root.

    Pathspecs resolve against git's working directory, and the executor need not run at
    the repository root, so a bare path would silently match nothing (or the wrong file)
    from a subdirectory. ``top`` anchors it at the root; ``literal`` keeps a file named
    ``*`` from matching every file. The global ``--literal-pathspecs`` would do the latter
    but disables this magic, so the guarantee lives on the one pathspec sent instead.
    """
    return f":(top,literal){path}"


def exclusion_reason(path: str, policy: DiffExclusionPolicy) -> Optional[str]:
    """Why a path git reported may not be shown, or ``None`` when it may.

    Takes the policy rather than the project so a caller checking hundreds of paths builds it once.
    """
    try:
        validate_duo_context_exclusions(path)
    except ToolException:
        return "excluded by Duo context exclusions"
    if not policy.is_allowed(path):
        return "excluded by project policy"
    return None


def path_is_excluded(path: str, policy: DiffExclusionPolicy) -> bool:
    return exclusion_reason(path, policy) is not None


def validate_repo_path(path: str, project: Any = None) -> str:
    """Return *path* normalised, or raise if it escapes the repo, is denylisted or is excluded.

    Normalised means forward slashes and no leading ``./``: the denylist check normalises
    the same way internally, and git gets the returned string verbatim inside a literal
    pathspec, where ``./app/a.rb`` matches nothing. ``validate_duo_context_exclusions``
    covers traversal and the secrets denylist. The leading-dash and absolute-path checks
    are this module's own: a path becomes an argv element, and although it always follows
    ``--``, a caller that forgets the separator must not be one typo away from passing an
    option.
    """
    path = path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]

    if not path:
        raise ToolException("Error: path must not be empty")

    if path.startswith("/") or path.startswith("-"):
        raise ToolException(
            f"Error: {path!r} must be a repository-relative path, spelled as it appears "
            "in the diff"
        )

    validate_duo_context_exclusions(path)

    if not DiffExclusionPolicy(project).is_allowed(path):
        raise ToolException(f"Error: {path!r} is excluded by project policy")

    return path


class ResolvedCommit(NamedTuple):
    sha: str
    shallow: bool


_UNTRACKED_WORKSPACE_MESSAGE = (
    "Error: the workspace folder is not a tracked directory of a git repository, so "
    "local git history tools are unavailable here."
)


class GitTooOldError(ToolException):
    """The checkout's git is below ``GIT_FLOOR`` or could not be probed; no command was sent."""


class ReadOnlyGitTool(DuoBaseTool):
    """Base for tools that read the local checkout with git and never write to it."""

    allowed_git_subcommands: ClassVar[frozenset[str]] = frozenset()
    _git_floor_met: bool = PrivateAttr(default=False)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # At class creation, not at call time: a subcommand that should never have been
        # reachable must fail the build, not the request that happens to use it.
        escaping = cls.allowed_git_subcommands - READ_ONLY_GIT_SUBCOMMANDS
        if escaping:
            raise TypeError(
                f"{cls.__name__} declares git subcommands outside "
                f"READ_ONLY_GIT_SUBCOMMANDS: {sorted(escaping)}"
            )

    async def _git(self, *argv: str) -> str:
        if not argv or argv[0] not in self.allowed_git_subcommands:
            raise ToolException(
                f"Error: git {argv[0] if argv else ''} is not a read-only command "
                f"available to {self.name}"
            )

        subcommand, *options = argv
        flags = options[: options.index("--")] if "--" in options else options
        if any(_asks_for_signature(option) for option in flags):
            raise ToolException(
                f"Error: {self.name} must not ask git for signature status"
            )
        for option in options:
            if _undoes_a_guarantee(option):
                raise ToolException(
                    f"Error: {self.name} must not pass git {option!r}; it undoes a "
                    "guarantee of this boundary"
                )
        if subcommand in _DIFF_SUBCOMMANDS:
            options = [*DIFF_SAFETY_FLAGS, *options]

        if self.metadata is None:
            raise ToolException(f"Error: {self.name} has no executor to run git with")

        await self._require_git_floor(self.metadata)

        # A result cap is not a work cap: `log -S` with few matches walks all history.
        raw = await _execute_action(
            self.metadata,
            run_command_action(
                "git",
                [*GIT_GLOBAL_FLAGS, subcommand, *options],
                resolve_command_timeout(),
            ),
        )

        return _unwrap_command_output(raw, argv[0])

    async def _require_git_floor(self, metadata: dict[str, Any]) -> None:
        """Refuse a git older than ``GIT_FLOOR``, once per instance, before any command runs.

        The lazy-fetch guard is only real from 2.29, so below the floor a poisoned partial clone would run its own
        commands from a tool that never prompts. An answer that does not parse is refused too.
        """
        if self._git_floor_met:
            return
        raw = await _execute_action(
            metadata,
            run_command_action(
                "git", [*GIT_GLOBAL_FLAGS, "version"], resolve_command_timeout()
            ),
        )
        floor = ".".join(map(str, GIT_FLOOR))
        try:
            reported = _unwrap_command_output(raw, "version").strip()
        except ToolException as exc:
            raise GitTooOldError(
                f"Error: {self.name} needs git {floor} or later and could not check "
                f"this checkout's ({exc})"
            ) from exc
        found = _GIT_VERSION_RE.match(reported)
        if found is None or tuple(map(int, found.groups())) < GIT_FLOOR:
            raise GitTooOldError(
                f"Error: {self.name} needs git {floor} or later; this checkout's git "
                f"reports {reported[:40]!r}. Update git to review local history."
            )
        self._git_floor_met = True

    async def _resolve_commit(self, ref: Optional[str]) -> ResolvedCommit:
        """Resolve *ref* (or HEAD) to a full SHA, the only place user text reaches git.

        Every other command receives the hex output of this call, so an argument that
        survived ``validate_ref`` still cannot influence a later argv position.

        The same call reports whether the clone is shallow, and where the executor's
        directory sits in the repository git discovered. GitLab CI's default
        ``GIT_DEPTH`` makes the former true in ambient executors, and truncated history
        reads exactly like an absent commit, so every tool needs the answer; riding on
        this ``rev-parse`` saves each of them a round trip to the user's machine. The
        latter feeds ``_require_tracked_workspace``.
        """
        validated = validate_ref(ref) if ref is not None else "HEAD"

        try:
            raw = await self._git(
                "rev-parse",
                "--show-prefix",
                "--is-shallow-repository",
                "--verify",
                f"{validated}^{{commit}}",
            )
        except GitTooOldError:
            raise
        except ToolException as exc:
            # An unknown ref exits 128; the generic envelope error would otherwise win.
            raise ToolException(
                f"Error: could not resolve {validated!r} to a commit in this checkout "
                f"({exc})"
            ) from exc

        # One line per option, in option order; the prefix line is empty at the root.
        lines = raw.removesuffix("\n").split("\n")
        if len(lines) != 3 or not _FULL_SHA_RE.match(lines[2]):
            raise ToolException(
                f"Error: could not resolve {validated!r} to a commit in this checkout"
            )
        prefix, probe, resolved = lines

        await self._require_tracked_workspace(prefix)

        # Anything but an explicit `false` counts as shallow: claiming complete history we
        # cannot vouch for is the worse error.
        return ResolvedCommit(sha=resolved, shallow=probe != "false")

    async def _require_tracked_workspace(self, prefix: str) -> None:
        """Refuse to read a repository the executor's directory merely sits inside.

        ``--show-prefix`` is empty at a repository root, and inside a nested repository,
        where discovery stops. It is non-empty for a subdirectory, which is a legitimate
        place to run from only when the repository tracks it. ``HEAD:<prefix>`` with the
        trailing slash git emits resolves only when the entry at that path is a tree, so
        an untracked, ignored or not-yet-committed folder fails closed, and so does a
        blob, symlink or submodule entry of the same name. The prefix is git's output
        and always follows ``HEAD:``, so it cannot be option-shaped, but the message
        stays fixed so that nothing about the enclosing repository travels back.
        """
        if not prefix:
            return
        if "\0" in prefix:
            raise ToolException(_UNTRACKED_WORKSPACE_MESSAGE)

        try:
            tree = await self._git("rev-parse", "--verify", "--quiet", f"HEAD:{prefix}")
        except ToolException:
            raise ToolException(_UNTRACKED_WORKSPACE_MESSAGE) from None
        if not _FULL_SHA_RE.match(tree.strip()):
            raise ToolException(_UNTRACKED_WORKSPACE_MESSAGE)


# The two lines a refused lazy fetch leaves behind: the fetch child's own, which is always
# present, and the parent's, which is absent when the parent died of SIGPIPE instead (see
# the module docstring). Anchored so that either phrase inside a commit message or path
# cannot hide the real error behind the partial-clone one.
_REFUSED_LAZY_FETCH = re.compile(
    r"^fatal: fetch\.parallel cannot be negative$"
    r"|^(?:fatal|error): could not fetch \S+ from promisor remote$",
    re.MULTILINE,
)

# Git prefixes every line it prints about a failure. Anything else in a failed command's
# merged output is stdout it wrote before dying, which can be a patch that nothing has
# filtered yet: `show` on a commit with one corrupt blob still prints the other files.
_DIAGNOSTIC_LINE = re.compile(r"^(?:fatal|error|warning|usage|hint): .*$", re.MULTILINE)
_MAX_DIAGNOSTIC_LINES = 5

_PARTIAL_CLONE_MESSAGE = (
    "Error: this checkout is a partial clone and git {subcommand} needs objects it does "
    "not have locally. Fetching them on demand is disabled here because it would run "
    "programs named in the repository's own config. Retrying will not help; use the "
    "GitLab API tools for this history or ask the user for a full clone."
)


# Git rejects the split `--format <fmt>` form itself; `-S` values are data, not a format.
def _asks_for_signature(option: str) -> bool:
    if option == "--show-signature":
        return True
    return option.startswith(("--format=", "--pretty")) and "%G" in option


def _undoes_a_guarantee(option: str) -> bool:
    name = option.partition("=")[0]
    return name in _GUARANTEE_UNDOING_OPTIONS


def _diagnostics(body: str) -> str:
    if not body.strip():
        return "(no output)"
    lines = _DIAGNOSTIC_LINE.findall(body)
    if not lines:
        return "(no diagnostic in the output)"
    # Git prints its hints and warnings first and the fatal line last, so keep the tail.
    if len(lines) > _MAX_DIAGNOSTIC_LINES:
        extra = len(lines) - _MAX_DIAGNOSTIC_LINES
        lines = [f"... {extra} earlier lines omitted", *lines[-_MAX_DIAGNOSTIC_LINES:]]
    return "\n".join(lines)


def _unwrap_command_output(raw: str, subcommand: str) -> str:
    """Strip the executor's exit-code header, raising when the command actually failed.

    The action reports success whatever git exited with, so the header is the only signal
    that the body is an error rather than output. A failure relays only git's own
    diagnostic lines, never the output that preceded them. A response over the executor's size limit
    puts its notice *before* the header; that notice is kept on the returned body so
    ``executor_truncated`` still sees it. An executor that returns the output bare is
    tolerated, so the tools work against both.
    """
    text = strip_size_limit_notice(raw)
    notice = raw[: len(raw) - len(text)]

    header = EXIT_CODE_HEADER.match(text)
    if header is None:
        return raw

    body = text[header.end() :]
    code = header.group(1)
    if code != "0":
        if _REFUSED_LAZY_FETCH.search(body):
            raise ToolException(_PARTIAL_CLONE_MESSAGE.format(subcommand=subcommand))
        detail = _diagnostics(body)
        raise ToolException(f"Error: git {subcommand} exited with {code}: {detail}")

    return notice + body
