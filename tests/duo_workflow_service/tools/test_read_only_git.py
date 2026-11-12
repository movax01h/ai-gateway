import os
import shutil
import subprocess
from pathlib import Path
from typing import ClassVar, List, Optional
from unittest.mock import patch

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.tools.read_only_git import (
    _GIT_TIMEOUT_SECONDS,
    DIFF_SAFETY_FLAGS,
    GIT_GLOBAL_FLAGS,
    READ_ONLY_GIT_SUBCOMMANDS,
    SHA_RE,
    ReadOnlyGitTool,
    output_was_truncated,
    validate_ref,
    validate_repo_path,
)

RESOLVED = "1111111111111111111111111111111111111111"

# Every git subcommand that can write to the repository, the index, the filesystem or the
# network. Asserted against the allowlist here rather than in each tool's test file, so
# the family is checked once and a later addition cannot slip past a tool-local test.
WRITING_SUBCOMMANDS = frozenset(
    {
        "add",
        "am",
        "apply",
        "checkout",
        "cherry-pick",
        "clean",
        "clone",
        "commit",
        "config",
        "fetch",
        "filter-branch",
        "gc",
        "init",
        "merge",
        "mv",
        "pull",
        "push",
        "rebase",
        "remote",
        "reset",
        "restore",
        "revert",
        "rm",
        "stash",
        "submodule",
        "switch",
        "tag",
        "update-index",
        "update-ref",
        "worktree",
        "write-tree",
    }
)


class _Probe(ReadOnlyGitTool):
    name: str = "probe"
    description: str = "test double"
    allowed_git_subcommands: ClassVar[frozenset[str]] = frozenset(
        {"rev-parse", "log", "show"}
    )

    async def _execute(self, **kwargs) -> str:
        return ""


class _Recorder:
    def __init__(self, output: str = ""):
        self.output = output
        self.argvs: List[List[str]] = []
        self.timeouts: List[Optional[int]] = []

    def __call__(self, _metadata, action):
        self.argvs.append(list(action.runCommand.arguments))
        self.timeouts.append(
            action.runCommand.timeout if action.runCommand.HasField("timeout") else None
        )
        return self.output


@pytest.fixture(name="probe")
def probe_fixture():
    return _Probe(metadata={"project": None})


def test_the_allowlist_contains_no_writing_subcommand():
    assert READ_ONLY_GIT_SUBCOMMANDS & WRITING_SUBCOMMANDS == frozenset()


@pytest.mark.parametrize("subcommand", ["blame", "status", "diff"])
def test_a_working_tree_reader_is_not_reachable(subcommand):
    """Each is read-only, but converts working-tree content through configured clean/process drivers.

    Re-adding a name without restoring that neutralisation would be arbitrary execution from a tool that never prompts.
    """
    assert subcommand not in READ_ONLY_GIT_SUBCOMMANDS


def test_a_tool_cannot_declare_a_subcommand_outside_the_superset():
    with pytest.raises(TypeError, match="outside READ_ONLY_GIT_SUBCOMMANDS.*'push'"):

        class _Escaping(ReadOnlyGitTool):
            name: str = "escaping"
            description: str = "test double"
            allowed_git_subcommands: ClassVar[frozenset[str]] = frozenset({"push"})

            async def _execute(self, **kwargs) -> str:
                return ""


@pytest.mark.asyncio
async def test_a_subcommand_the_tool_did_not_declare_is_refused(probe):
    recorder = _Recorder()

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        # `for-each-ref` is read-only and in the superset, but this tool never asked for it.
        with pytest.raises(ToolException, match="not a read-only command available"):
            await probe._git("for-each-ref")

    assert recorder.argvs == []


@pytest.mark.asyncio
async def test_every_action_carries_the_global_flags_first(probe):
    recorder = _Recorder()

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        await probe._git("log", "-n1")

    assert recorder.argvs == [[*GIT_GLOBAL_FLAGS, "log", *DIFF_SAFETY_FLAGS, "-n1"]]
    assert recorder.argvs[0][0] == "--no-optional-locks"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("subcommand", "injected"), [("log", True), ("show", True), ("rev-parse", False)]
)
async def test_textconv_is_disabled_at_the_egress_point(subcommand, injected):
    """`diff.<driver>.textconv` runs a configured program on tree-to-tree `log -p` and `show -p`, so the flag cannot be
    left to call sites.

    `rev-parse` echoes options it does not know back onto stdout, so it must not receive them.
    """

    class _Wide(ReadOnlyGitTool):
        name: str = "wide"
        description: str = "test double"
        allowed_git_subcommands: ClassVar[frozenset[str]] = frozenset(
            {"log", "show", "rev-parse"}
        )

        async def _execute(self, **kwargs) -> str:
            return ""

    recorder = _Recorder(output=f"{RESOLVED}\n")

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        await _Wide(metadata={"project": None})._git(subcommand, RESOLVED)

    argv = recorder.argvs[0][len(GIT_GLOBAL_FLAGS) :]
    assert argv[0] == subcommand
    assert (argv[1:3] == list(DIFF_SAFETY_FLAGS)) is injected
    assert argv[-1] == RESOLVED


@pytest.mark.asyncio
async def test_every_action_carries_a_timeout(probe):
    """A result cap is not a work cap: `git log -S` with few matches walks all history,
    and an action with no timeout carries no limit at all."""
    recorder = _Recorder()

    with (
        patch(
            "duo_workflow_service.tools.read_only_git._execute_action",
            side_effect=recorder,
        ),
        patch(
            "duo_workflow_service.tools.read_only_git.is_client_capable",
            return_value=True,
        ),
    ):
        await probe._git("log", "-n1")

    assert recorder.timeouts == [_GIT_TIMEOUT_SECONDS]


@pytest.mark.asyncio
async def test_an_older_executor_is_not_sent_a_field_it_cannot_parse(probe):
    recorder = _Recorder()

    with (
        patch(
            "duo_workflow_service.tools.read_only_git._execute_action",
            side_effect=recorder,
        ),
        patch(
            "duo_workflow_service.tools.read_only_git.is_client_capable",
            return_value=False,
        ),
    ):
        await probe._git("log", "-n1")

    assert recorder.timeouts == [None]


@pytest.mark.parametrize(
    "flag",
    [
        "--no-optional-locks",
        "--no-pager",
        "--literal-pathspecs",
        "diff.noprefix=false",
        "diff.mnemonicPrefix=false",
        "diff.suppressBlankEmpty=false",
        "diff.srcPrefix=a/",
        "diff.dstPrefix=b/",
        "core.fsmonitor=false",
        "diff.submodule=short",
        "fetch.parallel=-1",
        "log.showSignature=false",
        "gpg.program=false",
        "gpg.ssh.program=false",
        "gpg.x509.program=false",
        "log.follow=false",
        "core.warnAmbiguousRefs=false",
        "diff.relative=false",
        "core.quotePath=false",
        "protocol.allow=never",
    ],
)
def test_config_driven_output_is_pinned(flag):
    """Each of these blocks a poisoned .git/config turning a read into code execution or reshaping output the parsers
    and the exclusion filter rely on."""
    assert flag in GIT_GLOBAL_FLAGS


@pytest.mark.parametrize(
    "transport", ["file", "git", "ssh", "http", "https", "ftp", "ftps", "ext", "fd"]
)
def test_every_transport_is_refused_by_name(transport):
    """A repo-level `protocol.<name>.allow=always` beats the `protocol.allow` default, so the default alone leaves a
    poisoned config one line away from running its upload-pack."""
    assert f"protocol.{transport}.allow=never" in GIT_GLOBAL_FLAGS


@pytest.mark.asyncio
async def test_the_exit_code_envelope_is_stripped_before_parsing(probe):
    """The executor prefixes output with its exit code; left on, every parser here reads it as the opening line of
    data."""
    recorder = _Recorder(output=f"Exit code: 0\n{RESOLVED}\n")

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        assert await probe._resolve_commit("main") == RESOLVED


@pytest.mark.asyncio
async def test_a_failed_command_raises_with_gits_own_message(probe):
    """The action reports success even when the command failed, so a non-zero code is the only signal that the body is
    an error rather than output."""
    recorder = _Recorder(output="Exit code: 128\nfatal: not a git repository\n")

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        with pytest.raises(
            ToolException,
            match="git log exited with 128: fatal: not a git repository",
        ):
            await probe._git("log", "-n1")


@pytest.mark.asyncio
async def test_an_unknown_ref_gets_the_resolver_message_with_gits_reason(probe):
    """Rev-parse exits 128 for an unknown ref, which would otherwise surface as the generic envelope error."""
    recorder = _Recorder(output="Exit code: 128\nfatal: Needed a single revision\n")

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        with pytest.raises(
            ToolException,
            match=r"could not resolve 'nope' to a commit.*Needed a single revision",
        ):
            await probe._resolve_commit("nope")


@pytest.mark.asyncio
async def test_a_failed_commands_output_is_not_relayed_with_its_diagnostic(probe):
    """`show` on a commit with one corrupt blob prints the other files' diffs before it dies (verified against real
    git), and nothing downstream filters a failure body, so only git's own lines may travel."""
    recorder = _Recorder(
        output="Exit code: 128\n"
        "error: inflate: data stream error (incorrect header check)\n"
        "fatal: unable to read 975fbec8256d3e8a3797e7a3611380f27c49f4ac\n"
        "diff --git a/.env b/.env\n--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-secret=1\n+secret=2\n"
    )

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        with pytest.raises(ToolException) as excinfo:
            await probe._git("show", "--format=", RESOLVED)

    message = str(excinfo.value)
    assert "git show exited with 128" in message
    assert "error: inflate" in message
    assert "fatal: unable to read 975fbec8" in message
    assert "secret" not in message
    assert "diff --git" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("+secret=2\n", "(no diagnostic in the output)"),
        (
            "".join(f"error: line {i}\n" for i in range(8)),
            "... 3 earlier lines omitted\nerror: line 3\nerror: line 4\nerror: line 5\n"
            "error: line 6\nerror: line 7",
        ),
    ],
)
async def test_a_failure_body_without_or_with_many_diagnostics(probe, body, expected):
    recorder = _Recorder(output=f"Exit code: 1\n{body}")

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        with pytest.raises(ToolException) as excinfo:
            await probe._git("log")

    assert str(excinfo.value).endswith(expected)
    assert "secret" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_failure_with_no_output_still_names_the_code(probe):
    recorder = _Recorder(output="Exit code: 1\n")

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        with pytest.raises(ToolException, match=r"exited with 1: \(no output\)"):
            await probe._resolve_commit("main")


@pytest.mark.asyncio
async def test_a_size_limited_response_still_surfaces_the_failure(probe):
    """Over the executor's size limit the notice precedes the exit-code header, so a header match anchored at the start
    would miss a failed command and hand its stderr to the parser as output."""
    recorder = _Recorder(
        output=(
            "Maximum allowed size exceeded. Result: 5000000 bytes. "
            "Maximum allowed: 4194304 bytes\nExit code: 128\nfatal: bad object\n"
        )
    )

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        with pytest.raises(
            ToolException, match="git log exited with 128: fatal: bad object"
        ):
            await probe._git("log", "-p")


@pytest.mark.asyncio
async def test_a_size_limited_response_is_reported_as_truncated(probe):
    """The notice must survive unwrapping, or a sampled patch reads as a complete one."""
    notice = (
        "Maximum allowed size exceeded. Result: 5000000 bytes. "
        "Maximum allowed: 4194304 bytes\n"
    )
    recorder = _Recorder(output=f"{notice}Exit code: 0\ndiff --git a/a b/a\n")

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        body = await probe._git("log", "-p")

    assert output_was_truncated(body)
    assert "Exit code" not in body
    assert body.endswith("diff --git a/a b/a\n")


@pytest.mark.asyncio
async def test_a_bare_executor_response_is_tolerated(probe):
    """An executor that returns output without an envelope must keep working."""
    recorder = _Recorder(output=f"{RESOLVED}\n")

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        assert await probe._resolve_commit("main") == RESOLVED


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("diff --git a/a b/a\n[... output truncated: 2 of 9 KB shown]\nrest\n", True),
        ("diff --git a/a b/a\n+[... output truncated: not at line start]\n", False),
        ("clean output\n", False),
    ],
)
def test_executor_truncation_is_detected(raw, expected):
    """A diff line is always prefixed, so only the executor can start one with `[`."""
    assert output_was_truncated(raw) is expected


@pytest.mark.parametrize(
    "ref",
    [
        # `git log --output=<file>` is honoured and writes there, so an option-shaped
        # ref is a read-only tool writing to an attacker-chosen path.
        "--output=/tmp/pwned",
        "--upload-pack=touch /tmp/pwned",
        "-C/etc",
        "HEAD~3",
        "HEAD^",
        "a..b",
        "main b",
        "",
        ".hidden",
        # `$` would match before the trailing newline and let it into argv.
        "main\n",
        "abc123def\n",
    ],
)
def test_a_ref_that_is_not_a_plain_name_is_rejected(ref):
    with pytest.raises(ToolException, match="not a valid git ref name"):
        validate_ref(ref)


@pytest.mark.parametrize("ref", ["main", "origin/main", "v1.2.3", "abc123def", "HEAD"])
def test_a_plain_ref_name_is_accepted(ref):
    assert validate_ref(ref) == ref


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("", "must not be empty"),
        ("/etc/passwd", "must be a repository-relative path"),
        ("-C/etc/passwd", "must be a repository-relative path"),
        ("../../etc/passwd", "path traversal"),
        (".env", "Duo Context Exclusion"),
        (".git/config", "Duo Context Exclusion"),
        (".ssh/id_rsa", "Duo Context Exclusion"),
    ],
)
def test_a_path_that_escapes_or_is_denylisted_is_rejected(path, message):
    with pytest.raises(ToolException, match=message):
        validate_repo_path(path)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("./app/a.rb", "app/a.rb"),
        ("././app/a.rb", "app/a.rb"),
        ("app\\models\\a.rb", "app/models/a.rb"),
        ("app/a.rb", "app/a.rb"),
    ],
)
def test_a_path_is_normalised_to_what_git_will_match(path, expected):
    """The literal pathspec gets this string verbatim, and `./app/a.rb` matches nothing there."""
    assert validate_repo_path(path) == expected


@pytest.mark.parametrize("path", ["./", ".\\"])
def test_a_path_that_normalises_to_nothing_is_rejected(path):
    with pytest.raises(ToolException, match="must not be empty"):
        validate_repo_path(path)


def test_a_path_excluded_by_project_policy_is_rejected():
    project = {"exclusion_rules": ["secrets/*"]}

    with pytest.raises(ToolException, match="excluded by project policy"):
        validate_repo_path("secrets/keys.yml", project)

    assert validate_repo_path("app/a.rb", project) == "app/a.rb"


@pytest.mark.asyncio
async def test_a_ref_is_resolved_to_a_full_sha_before_any_other_command(probe):
    recorder = _Recorder(output=f"{RESOLVED}\n")

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        assert await probe._resolve_commit("main") == RESOLVED

    argv = recorder.argvs[0][len(GIT_GLOBAL_FLAGS) :]
    assert argv == ["rev-parse", "--verify", "main^{commit}"]


@pytest.mark.asyncio
async def test_resolution_defaults_to_head(probe):
    recorder = _Recorder(output=f"{RESOLVED}\n")

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        await probe._resolve_commit(None)

    assert recorder.argvs[0][-1] == "HEAD^{commit}"


@pytest.mark.asyncio
async def test_a_sha256_repository_resolves(probe):
    recorder = _Recorder(output=f"{'a' * 64}\n")

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        assert await probe._resolve_commit("main") == "a" * 64


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["not a sha\n", f"{'a' * 41}\n", f"{'a' * 63}\n"])
async def test_an_unresolvable_ref_raises_rather_than_passing_text_downstream(
    probe, output
):
    recorder = _Recorder(output=output)

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        with pytest.raises(ToolException, match="could not resolve 'main' to a commit"):
            await probe._resolve_commit("main")


@pytest.mark.asyncio
async def test_an_option_shaped_ref_never_reaches_git(probe):
    recorder = _Recorder()

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        with pytest.raises(ToolException, match="not a valid git ref name"):
            await probe._resolve_commit("--upload-pack=touch /tmp/pwned")

    assert recorder.argvs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("true\n", True),
        ("false\n", False),
        # An option echo from an unexpected git reads as shallow, not as complete.
        ("--is-shallow-repository\n", True),
        ("", True),
    ],
)
async def test_shallowness_is_read_from_git(probe, output, expected):
    recorder = _Recorder(output=output)

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        assert await probe._is_shallow() is expected


@pytest.mark.asyncio
async def test_an_unanswerable_shallowness_probe_assumes_truncation(probe):
    """Claiming complete history we cannot vouch for is the worse error."""

    def failing(_metadata, _action):
        raise ToolException("git too old")

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action", side_effect=failing
    ):
        assert await probe._is_shallow() is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "argv",
    [
        ("log", "--format=%h %G?"),
        ("log", "--pretty=format:%GS"),
        ("show", "--show-signature", "HEAD"),
    ],
)
async def test_a_signature_request_is_refused_before_dispatch(probe, argv):
    """The `gpg.*.program=false` pins are what keep a signature placeholder from running a program; this refusal is a
    tripwire so a tool author learns in tests that the pins are doing the work, rather than quietly relying on them."""
    recorder = _Recorder()

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        with pytest.raises(ToolException, match="signature status"):
            await probe._git(*argv)

    assert recorder.argvs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "argv",
    [
        ("log", "--ext-diff", "-p"),
        ("show", "--textconv", "HEAD"),
        ("show", "--submodule=diff", "HEAD"),
        ("log", "--output=/tmp/x"),
        ("log", "--output", "/tmp/x"),
        ("show", "--decorate-refs", "--", "--textconv", "HEAD"),
        ("show", "--decorate-refs", "--", "--ext-diff", "HEAD"),
        ("log", "--decorate-refs", "--", "--output=/tmp/x"),
    ],
)
async def test_an_option_that_undoes_a_guarantee_is_refused_before_dispatch(
    probe, argv
):
    """Git honours the last spelling of a flag, so a caller's `--textconv` would beat the `--no-textconv` that `_git`
    splices in first; `--submodule=diff` overrides the `diff.submodule` pin and `--output` writes a file.

    `--decorate-refs` takes `--` as its value, so git reads what follows as options: the check must not stop at `--`.
    """
    recorder = _Recorder()

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        with pytest.raises(ToolException, match="undoes a guarantee"):
            await probe._git(*argv)

    assert recorder.argvs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "argv",
    [
        ("log", "-S%G?", "HEAD"),
        ("log", "-S--textconv", "HEAD"),
        ("log", "--format=%h", "HEAD", "--", ":(top,literal)docs/100%Growth.md"),
    ],
)
async def test_a_placeholder_in_data_is_not_mistaken_for_a_signature_request(
    probe, argv
):
    """A pickaxe string or a filename may contain `%G` or spell an option; only a flag of its own makes git act on
    it."""
    recorder = _Recorder()

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        await probe._git(*argv)

    # `_git` splices the diff safety flags in after the subcommand; the options follow intact.
    assert recorder.argvs[0][-len(argv) + 1 :] == list(argv[1:])


@pytest.mark.asyncio
async def test_a_tool_without_executor_metadata_fails_before_dispatch():
    probe = _Probe(metadata=None)

    with patch("duo_workflow_service.tools.read_only_git._execute_action") as execute:
        with pytest.raises(ToolException, match="has no executor"):
            await probe._git("log")

    execute.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        "Exit code: 128\nfatal: transport 'ssh' not allowed\n"
        "fatal: could not fetch b77b4eb1 from promisor remote\n",
        "Exit code: 128\nfatal: fetch.parallel cannot be negative\n"
        "fatal: could not fetch b77b4eb1 from promisor remote\n",
        # The parent died of SIGPIPE before it could name the object; the child's line remains.
        "Exit code: -13\nfatal: fetch.parallel cannot be negative\n",
        # The same death as the Node executor reports it: a signal leaves no exit code.
        "Exit code: null\nfatal: fetch.parallel cannot be negative\n",
        # More diagnostics than the relay keeps still cannot hide the fetch line.
        "Exit code: 128\n"
        + "".join(f"hint: graft {i}\n" for i in range(8))
        + "fatal: fetch.parallel cannot be negative\n",
    ],
)
async def test_a_refused_lazy_fetch_is_explained_rather_than_echoed(probe, output):
    """Git's own messages name a config key, a transport or an object id, none of which tells the model what to do
    instead."""
    recorder = _Recorder(output=output)

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        with pytest.raises(ToolException) as excinfo:
            await probe._git("log", "-p")

    assert "partial clone and git log needs objects" in str(excinfo.value)
    assert "Retrying will not help" in str(excinfo.value)
    assert "b77b4eb1" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_the_promisor_phrase_inside_output_does_not_hide_the_real_error(probe):
    recorder = _Recorder(
        output="Exit code: 128\ncommit 1111111 fix: could not fetch x from promisor remote\n"
        "fatal: bad object 2222222\n"
    )

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=recorder,
    ):
        with pytest.raises(ToolException, match="exited with 128") as excinfo:
            await probe._git("log", "-p")

    assert "bad object" in str(excinfo.value)


# The fixtures and the "without the pins" proofs must not depend on the developer's own
# `commit.gpgsign`, `core.hooksPath` or similar.
_GIT_ENV = {
    "PATH": os.pathsep.join(
        [str(Path(shutil.which("git") or "/usr/bin/git").parent), os.defpath]
    ),
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(cwd: Path, *argv: str) -> None:
    subprocess.run(
        ["git", *argv], cwd=cwd, check=True, capture_output=True, env=_GIT_ENV
    )


def _partial_clone(tmp_path: Path) -> Path:
    """A `--filter=blob:none` clone whose blobs are all still on its promisor remote."""
    identity = ("-c", "user.email=t@example.com", "-c", "user.name=T")
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    (source / "a.rb").write_text("old\n")
    _git(source, "add", "a.rb")
    _git(source, *identity, "commit", "-qm", "one")
    (source / "a.rb").write_text("new\n")
    _git(source, *identity, "commit", "-qam", "two")
    origin = tmp_path / "origin.git"
    _git(source, "clone", "-q", "--bare", ".", str(origin))
    _git(origin, "config", "uploadpack.allowFilter", "true")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", "--filter=blob:none", f"file://{origin}", str(clone))
    _git(clone, "config", "remote.origin.url", str(origin))
    return clone


@pytest.mark.asyncio
async def test_a_partial_clone_cannot_run_its_promisor_transport(probe, tmp_path: Path):
    """A missing blob makes git fetch it from the promisor remote, and that fetch runs whatever the repo config names as
    the remote's upload-pack.

    The pins fail the fetch before that happens, even when the same config re-allows the protocol.
    """
    marker = tmp_path / "ran"
    helper = tmp_path / "upload-pack.sh"
    helper.write_text(f'#!/bin/sh\ntouch "{marker}"\nexec git-upload-pack "$@"\n')
    helper.chmod(0o755)
    clone = _partial_clone(tmp_path)
    _git(clone, "config", "remote.origin.uploadpack", str(helper))
    _git(clone, "config", "protocol.file.allow", "always")

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=_envelope_executor(clone),
    ):
        with pytest.raises(ToolException, match="checkout is a partial clone"):
            await probe._git("log", "--format=%h", "-p")

    assert not marker.exists()
    # The same command without the pins proves the fixture really does lazy-fetch.
    subprocess.run(
        ["git", "log", "--format=%h", "-p"],
        cwd=clone,
        capture_output=True,
        env=_GIT_ENV,
        check=False,
    )
    assert marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("argv", [("show", "HEAD"), ("log", "-S", "new", "--oneline")])
async def test_a_partial_clone_cannot_run_a_helper_its_own_config_defines(
    probe, tmp_path: Path, argv
):
    """A repo config can allow a protocol name no pin anticipates and supply the helper itself as an
    `alias.remote-<name>` shell command, since git resolves a non-builtin helper through aliases.

    The lazy fetch is a child `git fetch`, so a pin that kills it while reading config stops the helper from ever being
    chosen; the transport pins cannot, because the name is unknowable.
    """
    marker = tmp_path / "ran"
    clone = _partial_clone(tmp_path)
    _git(clone, "config", "remote.origin.url", "pwn::x")
    _git(clone, "config", "protocol.pwn.allow", "always")
    _git(clone, "config", "alias.remote-pwn", f'!touch "{marker}"')

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=_envelope_executor(clone),
    ):
        with pytest.raises(ToolException, match="checkout is a partial clone"):
            await probe._git(*argv)

    assert not marker.exists()
    # Without the pins the repo-defined helper really runs.
    subprocess.run(
        ["git", *argv], cwd=clone, capture_output=True, env=_GIT_ENV, check=False
    )
    assert marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("argv", [("show", "HEAD"), ("log", "-p", "-1", "HEAD")])
async def test_no_configured_diff_driver_runs_even_inside_a_submodule(
    probe, tmp_path: Path, argv
):
    """`--no-ext-diff --no-textconv` are options, not config, so they never reach the child `git diff` that
    `diff.submodule=diff` spawns for a submodule bump; pinning that setting keeps the child from existing at all."""
    markers = tmp_path / "markers"
    markers.mkdir()
    driver = tmp_path / "driver.sh"
    driver.write_text(f'#!/bin/sh\ntouch "{markers}/$1"\n')
    driver.chmod(0o755)
    identity = ("-c", "user.email=t@example.com", "-c", "user.name=T")

    sub = tmp_path / "sub"
    sub.mkdir()
    _git(sub, "init", "-q")
    (sub / "f.rb").write_text("a\n")
    (sub / ".gitattributes").write_text("*.rb diff=tc\n")
    _git(sub, "add", "-A")
    _git(sub, *identity, "commit", "-qm", "one")

    parent = tmp_path / "parent"
    parent.mkdir()
    _git(parent, "init", "-q")
    (parent / "p.rb").write_text("p\n")
    (parent / ".gitattributes").write_text("*.rb diff=tc\n")
    _git(parent, "add", "-A")
    _git(parent, *identity, "commit", "-qm", "init")
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(sub),
        "sub",
    )
    _git(parent, *identity, "commit", "-qm", "add sub")
    (parent / "sub" / "f.rb").write_text("b\n")
    _git(parent / "sub", *identity, "commit", "-qam", "two")
    (parent / "p.rb").write_text("q\n")
    _git(parent, *identity, "commit", "-qam", "bump")

    _git(parent, "config", "diff.submodule", "diff")
    sub_config = str(parent / ".git" / "modules" / "sub" / "config")
    for scope, name in ((), "parent"), (("--file", sub_config), "sub"):
        _git(parent, "config", *scope, "diff.external", f"{driver} {name}-external")
        _git(parent, "config", *scope, "diff.tc.textconv", f"{driver} {name}-textconv")

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=_envelope_executor(parent),
    ):
        await probe._git(*argv)

    assert not list(markers.iterdir())
    # Without the boundary the same command runs both repos' drivers, so the fixture is live.
    subprocess.run(
        ["git", *argv], cwd=parent, capture_output=True, env=_GIT_ENV, check=False
    )
    ran = {p.name for p in markers.iterdir()}
    assert "sub-external" in ran
    assert any(name.startswith("parent-") for name in ran)


def test_no_configured_signature_program_runs(tmp_path: Path):
    """A commit can carry any bytes as its signature, and `%G?` hands them to the program the repo config names for that
    format; `gpg.program=false` and its ssh/x509 siblings keep every one of them from being that program."""
    markers = tmp_path / "markers"
    markers.mkdir()
    driver = tmp_path / "driver.sh"
    driver.write_text(f'#!/bin/sh\ntouch "{markers}/$(basename "$0")"\nexit 1\n')
    driver.chmod(0o755)
    programs = {}
    for fmt in ("gpg", "ssh-keygen", "gpgsm"):
        programs[fmt] = tmp_path / fmt
        programs[fmt].write_text(driver.read_text())
        programs[fmt].chmod(0o755)
    signers = tmp_path / "signers"
    signers.write_text("")

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(
        repo,
        "-c",
        "user.email=t@example.com",
        "-c",
        "user.name=T",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "base",
    )
    tree = _read(repo, "rev-parse", "HEAD^{tree}")
    parent = _read(repo, "rev-parse", "HEAD")
    for armour in ("PGP SIGNATURE", "SSH SIGNATURE", "SIGNED MESSAGE"):
        body = (
            f"tree {tree}\nparent {parent}\n"
            "author T <t@example.com> 1700000000 +0000\n"
            "committer T <t@example.com> 1700000000 +0000\n"
            f"gpgsig -----BEGIN {armour}-----\n forged\n -----END {armour}-----\n\nforged\n"
        )
        (tmp_path / "commit.txt").write_text(body)
        parent = _read(
            repo, "hash-object", "-t", "commit", "-w", str(tmp_path / "commit.txt")
        )
    _git(repo, "update-ref", "HEAD", parent)
    _git(repo, "config", "gpg.program", str(programs["gpg"]))
    _git(repo, "config", "gpg.ssh.program", str(programs["ssh-keygen"]))
    _git(repo, "config", "gpg.x509.program", str(programs["gpgsm"]))
    _git(repo, "config", "gpg.ssh.allowedSignersFile", str(signers))

    argv = ["log", "-3", "--format=%G?"]
    subprocess.run(
        ["git", *GIT_GLOBAL_FLAGS, *argv],
        cwd=repo,
        capture_output=True,
        env=_GIT_ENV,
        check=False,
    )

    assert not list(markers.iterdir())
    # Without the pins the same command runs all three programs, so the fixture is live.
    subprocess.run(
        ["git", *argv], cwd=repo, capture_output=True, env=_GIT_ENV, check=False
    )
    assert {p.name for p in markers.iterdir()} == {"gpg", "ssh-keygen", "gpgsm"}


def _read(cwd: Path, *argv: str) -> str:
    return subprocess.run(
        ["git", *argv],
        cwd=cwd,
        check=True,
        capture_output=True,
        env=_GIT_ENV,
        text=True,
    ).stdout.strip()


def _envelope_executor(repo: Path):
    def execute(_metadata, action):
        result = subprocess.run(
            ["git", *action.runCommand.arguments],
            cwd=repo,
            capture_output=True,
            env=_GIT_ENV,
            text=True,
            check=False,
        )
        return f"Exit code: {result.returncode}\n{result.stdout}{result.stderr}"

    return execute


@pytest.mark.asyncio
async def test_a_real_sha256_repository_is_usable(probe, tmp_path: Path):
    repo = tmp_path / "sha256"
    repo.mkdir()
    _git(repo, "init", "-q", "--object-format=sha256")
    _git(
        repo,
        "-c",
        "user.email=t@example.com",
        "-c",
        "user.name=T",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "one",
    )

    with patch(
        "duo_workflow_service.tools.read_only_git._execute_action",
        side_effect=_envelope_executor(repo),
    ):
        resolved = await probe._resolve_commit(None)

    assert len(resolved) == 64
    assert SHA_RE.match(resolved)
