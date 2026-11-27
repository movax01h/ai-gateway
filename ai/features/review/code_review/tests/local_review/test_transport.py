"""The executor boundary: argv, the read-only allowlist and the exit-code envelope."""

from typing import List
from unittest.mock import patch

import pytest
from langchain_core.tools import ToolException

from ai.features.review.code_review.components.custom_instructions import (
    INSTRUCTIONS_FILE_PATH,
)
from ai.features.review.code_review.components.local_review.transport import (
    ALLOWED_GIT_SUBCOMMANDS,
    DIFF_FLAGS,
    GIT_GLOBAL_FLAGS,
    STATUS_FLAGS,
    GitCommandError,
    GitRunner,
    _filter_driver_names,
    is_denylisted,
)
from ai.features.review.code_review.tests.conftest import (
    APP_DIFF,
    BASE_SHA,
    DIFF,
    FakeGit,
    _strip_global_flags,
    numstat,
    responses,
    run_tool,
)
from duo_workflow_service.bl_security.executor_output import executor_truncated
from duo_workflow_service.tools.read_only_git import (
    GIT_GLOBAL_FLAGS as READ_ONLY_GIT_GLOBAL_FLAGS,
)


def test_pathspecs_are_passed_literally():
    """Structural guard: every invocation disables pathspec magic, since paths here are cwd-relative data."""
    assert "--literal-pathspecs" in GIT_GLOBAL_FLAGS


def test_config_driven_program_execution_is_disabled():
    """`git status` runs `core.fsmonitor` as a program, and this tool never asks for approval."""
    flags = list(GIT_GLOBAL_FLAGS)

    assert "core.fsmonitor=false" in flags
    assert flags[flags.index("core.fsmonitor=false") - 1] == "-c"


def test_diff_prefixes_are_pinned_against_user_config():
    """Git >= 2.45 lets `diff.srcPrefix`/`dstPrefix` replace a/ and b/, which the parser strips."""
    flags = list(GIT_GLOBAL_FLAGS)

    assert "diff.srcPrefix=a/" in flags
    assert "diff.dstPrefix=b/" in flags


@pytest.mark.parametrize("flags", [DIFF_FLAGS, STATUS_FLAGS])
def test_submodules_are_skipped_on_the_command_line(flags):
    """Recursing runs each submodule's own filter drivers, which the superproject's config never named.

    `diff.ignoreSubmodules` would not do: `submodule.<name>.ignore` in the in-tree `.gitmodules` overrides it.
    """
    assert "--ignore-submodules=all" in flags


@pytest.mark.asyncio
async def test_denylisted_file_read_returns_nothing_without_an_action(metadata):
    fake = FakeGit(responses())
    git = GitRunner(metadata)

    with patch(
        "ai.features.review.code_review.components.local_review.transport._execute_action",
        side_effect=fake,
    ):
        assert await git.read_file(".env") is None

    assert fake.actions == []


@pytest.mark.asyncio
async def test_unreadable_file_is_not_an_error(metadata):
    """`read_file` swallows the action failure; the caller reports the file as excluded."""
    fake = FakeGit(responses())
    git = GitRunner(metadata)

    with patch(
        "ai.features.review.code_review.components.local_review.transport._execute_action",
        side_effect=fake,
    ):
        assert await git.read_file("gone.txt") is None


@pytest.mark.asyncio
async def test_a_write_subcommand_is_refused_before_it_reaches_the_executor(metadata):
    """The allowlist is asserted on the way out, so no later edit can make this tool write."""
    fake = FakeGit(responses())
    git = GitRunner(metadata)

    with patch(
        "ai.features.review.code_review.components.local_review.transport._execute_action",
        side_effect=fake,
    ):
        with pytest.raises(ToolException, match="not a read-only command"):
            await git.run("commit", "-m", "nope")

    assert fake.actions == []


PATCH = (
    "ai.features.review.code_review.components.local_review.transport._execute_action"
)


@pytest.mark.asyncio
async def test_no_subcommand_is_refused_before_it_reaches_the_executor(metadata):
    fake = FakeGit(responses())

    with patch(PATCH, side_effect=fake):
        with pytest.raises(ToolException, match="no subcommand"):
            await GitRunner(metadata).run()

    assert fake.actions == []


@pytest.mark.parametrize(
    "option", ["--output=/tmp/x", "--output", "--ext-diff", "--textconv", "--no-index"]
)
@pytest.mark.asyncio
async def test_an_option_that_writes_reads_outside_or_runs_a_program_is_refused(
    metadata, option
):
    """`--no-ext-diff` is positional, `--output` writes and `--no-index` leaves the repository, so the subcommand
    allowlist is not enough."""
    fake = FakeGit(responses())

    with patch(PATCH, side_effect=fake):
        with pytest.raises(ToolException, match="is not allowed"):
            await GitRunner(metadata).run("diff", option, "HEAD")

    assert fake.actions == []


@pytest.mark.parametrize(
    "argv",
    [
        ("diff", "--no-textconv", "--ignore-submodules=all", "HEAD"),
        ("diff", "--no-ext-diff", "--ignore-submodules=all", "HEAD"),
        ("diff", "--no-ext-diff", "--no-textconv", "HEAD"),
        ("status", "--porcelain", "-z"),
    ],
)
@pytest.mark.asyncio
async def test_a_diff_or_status_without_its_safety_flags_is_refused(metadata, argv):
    """The flags are defined once in `DIFF_FLAGS` and `STATUS_FLAGS`, but forgetting them at a call site is not
    an option: a configured `diff.external` or a submodule's filter driver would run."""
    fake = FakeGit(responses())

    with patch(PATCH, side_effect=fake):
        with pytest.raises(ToolException, match="must be run with"):
            await GitRunner(metadata).run(*argv)

    assert fake.actions == []


@pytest.mark.parametrize(
    "option",
    ["--ignore-submodules", "--ignore-submodules=none", "--no-ignore-submodules"],
)
@pytest.mark.asyncio
async def test_another_ignore_submodules_spelling_is_refused(metadata, option):
    """Git takes the last spelling, so any other one after the pin would recurse into submodules again."""
    fake = FakeGit(responses())

    with patch(PATCH, side_effect=fake):
        with pytest.raises(ToolException, match="is not allowed"):
            await GitRunner(metadata).run("status", *STATUS_FLAGS, option)

    assert fake.actions == []


@pytest.mark.asyncio
async def test_a_denied_option_is_refused_wherever_it_appears_in_argv(metadata):
    """Whether `--` ends the options is a per-subcommand fact, so the check does not lean on it."""
    fake = FakeGit(responses())

    with patch(PATCH, side_effect=fake):
        with pytest.raises(ToolException, match="is not allowed"):
            await GitRunner(metadata).run(
                "diff", "--src-prefix", "--", "--output=/tmp/x"
            )

    assert fake.actions == []


@pytest.mark.asyncio
async def test_a_paged_read_is_followed_to_the_end(metadata):
    """The executor pages reads at 2,000 lines and reports the first page as success."""
    pages = {
        0: "one\ntwo\n\n(Showing lines 1-2 of 3 total. Use offset=2 to continue reading.)",
        2: "three\n",
    }

    async def execute(_metadata, action):
        return pages[action.runReadFile.offset]

    with patch(PATCH, side_effect=execute):
        assert await GitRunner(metadata).read_file("big.txt") == "one\ntwo\nthree\n"


@pytest.mark.asyncio
async def test_a_read_that_cannot_be_completed_is_not_returned_as_a_prefix(metadata):
    """An executor that keeps serving the same page stalls; the caller sees no file, not part of one."""
    page = "one\n\n(Showing lines 1-1 of 9 total. Use offset=1 to continue reading.)"

    with patch(PATCH, return_value=page):
        assert await GitRunner(metadata).read_file("big.txt") is None


def test_empty_path_is_not_denylisted():
    assert is_denylisted("") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "argv",
    [
        ("config", "user.email", "attacker@example.com"),
        ("config", "--global", "core.pager", "sh -c id"),
        ("config", "--unset", "filter.lfs.clean"),
        ("config", "--get-regexp", "^core\\."),
    ],
)
async def test_config_is_allowed_only_as_the_read_only_filter_query(metadata, argv):
    """The allowlist matches on subcommand alone, and `git config` can write."""
    fake = FakeGit(responses())
    git = GitRunner(metadata)

    with patch(
        "ai.features.review.code_review.components.local_review.transport._execute_action",
        side_effect=fake,
    ):
        with pytest.raises(ToolException, match="read-only query"):
            await git.run(*argv)

    assert fake.actions == []


def test_driver_names_are_read_from_the_config_query():
    """A name may contain dots, so the key is split on the known suffixes, and spaces, so entries are NUL-terminated.

    Splitting on the first space would read `filter.my driver.clean` as `filter.my` and leave that driver live.
    """
    raw = (
        "filter.lfs.clean\ngit-lfs clean -- %f\0"
        "filter.my.team.filter.smudge\ncat\0"
        "filter.my driver.clean\ncat\0"
        "filter.ignored.unrelated\nvalue\0"
    )

    assert _filter_driver_names(raw) == {"lfs", "my.team.filter", "my driver"}


@pytest.mark.asyncio
async def test_a_non_zero_exit_is_not_read_as_data(metadata):
    """`run_command` reports a non-zero exit as a successful action, so the body carries it."""
    git = GitRunner(metadata)

    with patch(
        "ai.features.review.code_review.components.local_review.transport._execute_action",
        side_effect=lambda *_: "Exit code: 128\nfatal: not a git repository\n",
    ):
        with pytest.raises(ToolException, match="exited with 128"):
            await git.run("status", *STATUS_FLAGS)


@pytest.mark.asyncio
async def test_the_exit_code_envelope_is_stripped_before_parsing(metadata):
    """Every response arrives as `Exit code: 0\\n<output>`; unstripped it becomes the first datum."""
    git = GitRunner(metadata)

    with patch(
        "ai.features.review.code_review.components.local_review.transport._execute_action",
        side_effect=lambda *_: (
            "Exit code: 0\n1111111111111111111111111111111111111111\n"
        ),
    ):
        assert (
            await git.run("rev-parse", "HEAD")
        ) == "1111111111111111111111111111111111111111\n"


@pytest.mark.asyncio
async def test_no_configured_filters_is_not_an_error(metadata):
    """`--get-regexp` exits non-zero when nothing matches, which is the common case."""
    git = GitRunner(metadata)

    # A non-zero exit arrives as a *successful* action carrying the code in the body.
    with patch(
        "ai.features.review.code_review.components.local_review.transport._execute_action",
        side_effect=lambda *_: "Exit code: 1\n",
    ):
        assert await git.filter_drivers() == ()


@pytest.mark.asyncio
async def test_the_prefix_is_the_executor_directory_below_the_repository_root(metadata):
    """`status --porcelain` has no `--relative`, so its paths are rebased onto this."""
    git = GitRunner(metadata)

    with patch(
        "ai.features.review.code_review.components.local_review.transport._execute_action",
        side_effect=lambda *_: "Exit code: 0\nsub/deep/\n",
    ):
        assert await git.prefix() == "sub/deep/"


@pytest.mark.asyncio
async def test_every_configured_driver_is_emptied_by_name(metadata):
    """There is no global switch, so each driver is named and emptied individually."""
    git = GitRunner(metadata)
    config = (
        "filter.lfs.clean\ngit-lfs clean -- %f\0"
        "filter.lfs.required\ntrue\0"
        "filter.my.team.smudge\ncat\0"
    )
    fake = FakeGit(responses(config=config))

    with patch(PATCH, side_effect=fake):
        drivers = await git.filter_drivers()
        await git.run("status", *STATUS_FLAGS, filter_drivers=drivers)

    assert drivers == ("lfs", "my.team")
    overrides = _overrides(fake.raw_argvs[-1], "status")
    assert "filter.lfs.clean=" in overrides
    # `required` is cleared rather than emptied: an emptied command reads as a failed filter.
    assert "filter.lfs.required=false" in overrides
    assert "filter.my.team.clean=" in overrides


@pytest.mark.asyncio
async def test_a_driver_name_is_only_ever_a_filter_subsection(metadata):
    """Overrides follow the pins, so a raw `-c` here could re-enable fsmonitor; only a name gets through."""
    fake = FakeGit(responses())

    with patch(PATCH, side_effect=fake):
        await GitRunner(metadata).run(
            "status", *STATUS_FLAGS, filter_drivers=("core.fsmonitor",)
        )

    assert _overrides(fake.raw_argvs[-1], "status") == [
        "-c",
        "filter.core.fsmonitor.clean=",
        "-c",
        "filter.core.fsmonitor.smudge=",
        "-c",
        "filter.core.fsmonitor.process=",
        "-c",
        "filter.core.fsmonitor.required=false",
    ]


@pytest.mark.parametrize("name", ["a=b", "x.clean= -c core.fsmonitor=evil"])
@pytest.mark.asyncio
async def test_a_driver_that_cannot_be_disabled_with_c_fails_closed(metadata, name):
    """`-c` splits at the first `=`, so `filter.a=b.clean=` sets `filter.a` and leaves `a=b` live.

    A subsection may contain `=` and `.gitattributes` can name it as `filter=a=b`, so this is the same poisoned-config
    case the overrides exist for. Nothing reads the working tree once the driver cannot be neutralised.
    """
    fake = FakeGit(responses(config=f"filter.{name}.clean\ncat\0"))
    git = GitRunner(metadata)

    with patch(PATCH, side_effect=fake):
        drivers = await git.filter_drivers()
        with pytest.raises(ToolException, match="cannot be disabled with -c"):
            await git.run("status", *STATUS_FLAGS, filter_drivers=drivers)

    assert drivers == (name,)
    assert [argv[0] for argv in fake.git_argvs] == ["config"]


def _overrides(argv, subcommand):
    """The `-c` pairs `run` placed between the global flags and the subcommand."""
    return argv[len(GIT_GLOBAL_FLAGS) : argv.index(subcommand)]


@pytest.mark.asyncio
async def test_an_executor_that_does_not_wrap_output_still_works(metadata):
    """The envelope is tolerated rather than required, so the tool works either way."""
    git = GitRunner(metadata)

    with patch(
        "ai.features.review.code_review.components.local_review.transport._execute_action",
        side_effect=lambda *_: "no envelope here\n",
    ):
        assert await git.run("rev-parse", "HEAD") == "no envelope here\n"


@pytest.mark.asyncio
async def test_a_config_failure_other_than_no_match_is_not_swallowed(metadata):
    """Returning no overrides on any failure would leave every configured driver live.

    `--get-regexp` exits 1 when nothing matches, which is ordinary; a corrupt config exits 128.
    """
    git = GitRunner(metadata)

    with patch(
        "ai.features.review.code_review.components.local_review.transport._execute_action",
        side_effect=lambda *_: "Exit code: 128\nfatal: bad config line 1\n",
    ):
        with pytest.raises(ToolException, match="exited with 128"):
            await git.filter_drivers()


@pytest.mark.asyncio
async def test_a_transport_failure_is_not_read_as_no_filters(metadata):
    """An executor error is not an answer about which filters are configured."""
    git = GitRunner(metadata)

    with patch(
        "ai.features.review.code_review.components.local_review.transport._execute_action",
        side_effect=ToolException("Action error: executor unreachable"),
    ):
        with pytest.raises(ToolException, match="executor unreachable"):
            await git.filter_drivers()


@pytest.mark.parametrize(
    "listing",
    [
        "filter.lfs.clean\ngit-lfs clean -- %f\0"
        "\n\n[... output truncated: showing first and last 25KB of 80.0KB ...]\n\n"
        "filter.zzz.smudge\ncat\0",
        "Maximum allowed size exceeded. Showing a sample.\nfilter.lfs.clean\ngit-lfs clean -- %f\0",
    ],
)
@pytest.mark.asyncio
async def test_a_truncated_filter_listing_fails_closed(metadata, listing):
    """The executor keeps the head and tail of a long listing, so a driver in the cut middle would stay live."""
    fake = FakeGit(responses(config=listing))

    with patch(PATCH, side_effect=fake):
        with pytest.raises(ToolException, match="too large to read in full"):
            await GitRunner(metadata).filter_drivers()

    assert [argv[0] for argv in fake.git_argvs] == ["config"]


def test_a_failure_relays_only_gits_diagnostic_lines():
    """Git flushes stdout before it dies, so a failed diff can carry hunks nothing has filtered yet."""
    error = GitCommandError("diff", "128", "diff --git a/f b/f\n+secret\nfatal: boom\n")

    assert str(error) == "Error: git diff exited with 128: fatal: boom"


def test_the_exit_code_is_carried_on_the_exception():
    """`filter_drivers` branches on the code, so it has to survive as data, not prose."""
    error = GitCommandError("config", "128", "fatal: bad config")

    assert error.code == "128"
    assert "exited with 128" in str(error)


def test_global_flags_are_the_shared_hardened_set():
    """One set of poisoned-config pins for every git boundary; these are the ones this package leans on."""
    assert (
        GIT_GLOBAL_FLAGS[: len(READ_ONLY_GIT_GLOBAL_FLAGS)]
        == READ_ONLY_GIT_GLOBAL_FLAGS
    )
    flags = list(GIT_GLOBAL_FLAGS)
    # `rev-parse` and `merge-base` output is read as a bare SHA, and the executor merges stderr into it.
    assert "core.warnAmbiguousRefs=false" in flags
    # A partial clone would otherwise fetch missing blobs through programs named in its own config.
    assert "fetch.parallel=-1" in flags


SIZE_LIMIT_NOTICE = "Maximum allowed size exceeded. Showing a sample.\n"


@pytest.mark.asyncio
async def test_a_size_limit_notice_does_not_hide_the_exit_code(metadata):
    """A response over the executor's size limit puts its notice before the header."""
    git = GitRunner(metadata)

    with patch(
        PATCH,
        side_effect=lambda *_: f"{SIZE_LIMIT_NOTICE}Exit code: 128\nfatal: boom\n",
    ):
        with pytest.raises(GitCommandError, match="exited with 128"):
            await git.run("diff", *DIFF_FLAGS)


@pytest.mark.asyncio
async def test_a_size_limit_notice_stays_on_successful_output(metadata):
    """Stripped to find the header, kept so `executor_truncated` still reports the cut."""
    git = GitRunner(metadata)

    with patch(
        PATCH, side_effect=lambda *_: f"{SIZE_LIMIT_NOTICE}Exit code: 0\nsome diff\n"
    ):
        output = await git.run("diff", *DIFF_FLAGS)

    assert output == f"{SIZE_LIMIT_NOTICE}some diff\n"
    assert executor_truncated(output)


@pytest.mark.asyncio
async def test_a_refused_lazy_fetch_is_reported_as_a_partial_clone(metadata):
    """The pins make the fetch child die on `fetch.parallel`; the user is told why rather than shown a config error."""
    git = GitRunner(metadata)
    body = (
        "fatal: fetch.parallel cannot be negative\n"
        "error: could not fetch 1234abcd from promisor remote\n"
    )

    with patch(PATCH, side_effect=lambda *_: f"Exit code: 128\n{body}"):
        with pytest.raises(ToolException, match="partial clone"):
            await git.run("diff", *DIFF_FLAGS)


LFS_CONFIG = (
    "filter.lfs.clean\ngit-lfs clean -- %f\0"
    "filter.lfs.smudge\ngit-lfs smudge -- %f\0"
    "filter.lfs.process\ngit-lfs filter-process\0"
    "filter.lfs.required\ntrue\0"
)


def _overrides_in(argv: List[str]) -> List[str]:
    """The `filter.*` overrides a raw git invocation carried, as `key=value` strings."""
    return [
        argv[i + 1]
        for i, token in enumerate(argv[:-1])
        if token == "-c" and argv[i + 1].startswith("filter.")
    ]


@pytest.mark.asyncio
async def test_never_writes_to_the_repository(metadata):
    """Every emitted git action is read-only.

    This is the tool's core guarantee.
    """
    fake = FakeGit(
        responses(
            status=numstat(" M app/a.rb", "?? scratch.txt"),
            **{
                f"{DIFF} --numstat -z {BASE_SHA}": numstat("1\t1\tapp/a.rb"),
                f"{DIFF} {BASE_SHA}": APP_DIFF,
            },
        ),
        files={"scratch.txt": "new line\n"},
    )

    await run_tool(fake, metadata, mode="working_tree")
    await run_tool(fake, metadata, mode="branch")
    await run_tool(fake, metadata, only_instructions=True)

    forbidden = {
        "add",
        "commit",
        "checkout",
        "switch",
        "stash",
        "reset",
        "restore",
        "write-tree",
        "update-index",
        "fetch",
        "push",
        "clean",
        "apply",
        "-N",
        "--intent-to-add",
    }
    for argv in fake.git_argvs:
        assert argv[0] in ALLOWED_GIT_SUBCOMMANDS, argv
        assert not forbidden.intersection(argv), argv

    # Without this flag, `status` and `diff` opportunistically rewrite the index.
    raw_argvs = [
        list(a.runCommand.arguments)
        for a in fake.actions
        if a.WhichOneof("action") == "runCommand"
    ]
    assert raw_argvs and all(argv[0] == "--no-optional-locks" for argv in raw_argvs)


@pytest.mark.asyncio
async def test_git_argv_is_never_split_across_the_proto_flags_field(metadata):
    """The executor concatenates `[...flags, ...args]`, so a split would reorder the argv.

    `-c key=value` is a pair of tokens, and reordering can separate them.
    """
    fake = FakeGit(responses())

    await run_tool(fake, metadata)

    commands = [a for a in fake.actions if a.WhichOneof("action") == "runCommand"]
    assert commands
    for action in commands:
        assert list(action.runCommand.flags) == [], action


@pytest.mark.asyncio
async def test_configured_filters_are_disabled_before_the_working_tree_is_read(
    metadata,
):
    """`--no-ext-diff`/`--no-textconv` do not cover clean/smudge/process drivers.

    A `filter` attribute makes git run the configured program to canonicalise working-tree
    content, which is arbitrary execution from a tool that never asks for approval.
    """
    fake = FakeGit(responses(**{"config": LFS_CONFIG}))

    await run_tool(fake, metadata, mode="working_tree")

    expected = [
        "filter.lfs.clean=",
        "filter.lfs.process=",
        "filter.lfs.required=false",
        "filter.lfs.smudge=",
    ]
    worktree_reading = [
        argv
        for argv in fake.raw_argvs
        if _strip_global_flags(argv)[0] in ("status", "diff", "check-attr")
    ]
    assert worktree_reading
    for argv in worktree_reading:
        assert sorted(_overrides_in(argv)) == expected, argv


@pytest.mark.asyncio
async def test_the_filter_query_runs_before_anything_reads_the_working_tree(metadata):
    """Overrides discovered after the first worktree read would be too late to matter."""
    fake = FakeGit(responses(**{"config": LFS_CONFIG}))

    await run_tool(fake, metadata, mode="working_tree")

    subcommands = [argv[0] for argv in fake.git_argvs]
    first_worktree_read = min(
        subcommands.index(name) for name in ("status", "diff") if name in subcommands
    )
    assert subcommands.index("config") < first_worktree_read


@pytest.mark.asyncio
async def test_a_repository_with_no_filters_adds_no_overrides(metadata):
    fake = FakeGit(responses())

    await run_tool(fake, metadata, mode="working_tree")

    assert all(_overrides_in(argv) == [] for argv in fake.raw_argvs)


@pytest.mark.asyncio
async def test_driver_names_containing_dots_are_handled(metadata):
    """`filter.my.driver.clean` names the driver `my.driver`, not `my`."""
    fake = FakeGit(responses(**{"config": "filter.my.driver.clean\ncat\0"}))

    await run_tool(fake, metadata, mode="working_tree")

    status = next(
        argv for argv in fake.raw_argvs if _strip_global_flags(argv)[0] == "status"
    )
    assert "filter.my.driver.clean=" in _overrides_in(status)


@pytest.mark.asyncio
async def test_a_required_filter_has_its_required_flag_cleared(metadata):
    """Git reads an emptied command as a filter that failed, not one that is absent.

    `git lfs install` sets `required = true`, so without clearing it every command aborts with "clean filter 'lfs'
    failed" and the review produces nothing at all.
    """
    fake = FakeGit(
        responses(
            **{
                "config": "filter.lfs.clean\ngit-lfs clean -- %f\0filter.lfs.required\ntrue\0"
            }
        )
    )

    await run_tool(fake, metadata, mode="working_tree")

    worktree_reading = [
        argv
        for argv in fake.raw_argvs
        if _strip_global_flags(argv)[0] in ("status", "diff", "check-attr")
    ]
    assert worktree_reading
    for argv in worktree_reading:
        assert sorted(_overrides_in(argv)) == [
            "filter.lfs.clean=",
            "filter.lfs.process=",
            "filter.lfs.required=false",
            "filter.lfs.smudge=",
        ], argv


@pytest.mark.asyncio
async def test_a_diff_whose_content_mentions_the_marker_is_still_reviewed(metadata):
    """The executor writes the marker on its own line; a diff line is always prefixed.

    Matching it anywhere excluded any file discussing truncation -- including this repository's own tests for it.
    """
    mentions = (
        "diff --git a/app/a.rb b/app/a.rb\n--- a/app/a.rb\n+++ b/app/a.rb\n"
        "@@ -1,1 +1,2 @@\n-a = 1\n"
        '+marker = "[... output truncated: showing first and last 25KB of 1.2MB ...]"\n'
    )
    fake = FakeGit(responses(**{f"{DIFF} {BASE_SHA} HEAD": mentions}))

    result = await run_tool(fake, metadata, mode="branch")

    assert '<file_diff filename="app/a.rb">' in result
    assert "diff too large" not in result


@pytest.mark.asyncio
async def test_the_instructions_file_is_read_despite_living_under_the_denylist(
    metadata,
):
    """`.gitlab/duo` is denylisted for general reads; the literal instructions path is the one exemption."""
    assert is_denylisted(INSTRUCTIONS_FILE_PATH)
    fake = FakeGit(responses(), files={INSTRUCTIONS_FILE_PATH: "---\n"})

    with patch(PATCH, side_effect=fake):
        assert await GitRunner(metadata).read_file(INSTRUCTIONS_FILE_PATH) == "---\n"
