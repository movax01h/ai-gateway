"""Parsing git's output, including reconciling its two coordinate systems below the root."""

from typing import Dict

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.tools.code_review.local_review.transport import (
    DIFF_FLAGS,
)
from tests.duo_workflow_service.tools.code_review.conftest import (
    BASE_SHA,
    DIFF,
    FakeGit,
    _file_diff,
    numstat,
    responses,
    run_tool,
)


@pytest.mark.asyncio
async def test_renamed_entry_in_status_consumes_its_origin_path(metadata):
    """A rename's origin is a second field; misreading it shifts every entry after it."""
    fake = FakeGit(
        responses(status=numstat("R  new.rb", "old.rb", " M after.rb")),
    )

    result = await run_tool(fake, metadata, mode="branch")

    # `after.rb` is only reported as uncommitted if the origin field was consumed correctly.
    assert "- after.rb" in result
    assert "- new.rb" in result


@pytest.mark.asyncio
async def test_short_status_entry_is_ignored(metadata):
    fake = FakeGit(responses(status=numstat("?", " M real.rb")))

    result = await run_tool(fake, metadata, mode="branch")

    assert "- real.rb" in result


@pytest.mark.asyncio
async def test_numstat_entry_without_counts_is_ignored(metadata):
    fake = FakeGit(
        responses(
            **{
                f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat(
                    "garbage", "1\t1\tapp/a.rb"
                )
            }
        )
    )

    result = await run_tool(fake, metadata, mode="branch")

    assert '<file_diff filename="app/a.rb">' in result


@pytest.mark.asyncio
async def test_truncated_rename_numstat_stops_cleanly(metadata):
    """A rename's paths are two more fields; a stream that ends first must not IndexError."""
    fake = FakeGit(
        responses(
            **{f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat("1\t1\t", "only.rb")}
        )
    )

    with pytest.raises(ToolException, match="nothing to review"):
        await run_tool(fake, metadata, mode="branch")


@pytest.mark.asyncio
async def test_filenames_containing_tabs_survive_numstat_parsing(metadata):
    """`--numstat -z` does not quote paths, so only the first two tabs are column separators."""
    fake = FakeGit(
        responses(
            **{
                f"{DIFF} --numstat -z {BASE_SHA}": numstat("1\t1\ttab\there.txt"),
                f"{DIFF} {BASE_SHA}": _file_diff("tab\there.txt", 1, "x"),
            }
        ),
    )

    result = await run_tool(fake, metadata)

    assert "- tab\there.txt" in result
    assert '<file_diff filename="tab\there.txt">' in result


@pytest.mark.asyncio
async def test_non_ascii_paths_match_their_numstat_entries(metadata):
    """`core.quotePath` octal-escapes UTF-8 bytes in diff headers; `--numstat -z` emits the path verbatim."""
    quoted_diff = (
        'diff --git "a/na\\303\\257ve.py" "b/na\\303\\257ve.py"\n'
        "index 111..222 100644\n"
        '--- "a/na\\303\\257ve.py"\n'
        '+++ "b/na\\303\\257ve.py"\n'
        "@@ -1,1 +1,1 @@\n"
        "-a = 1\n"
        "+a = 2\n"
    )
    fake = FakeGit(
        responses(
            **{
                f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat("1\t1\tnaïve.py"),
                f"{DIFF} {BASE_SHA} HEAD": quoted_diff,
            }
        )
    )

    result = await run_tool(fake, metadata)

    assert '<file_diff filename="naïve.py">' in result
    assert "- naïve.py" in result
    assert "na\\303\\257ve" not in result


def nested(**overrides: str) -> Dict[str, str]:
    """Responses for an executor running in `sub/` of the repository.

    `diff --relative` reports and scopes to that directory; `status --porcelain` does not and keeps reporting from the
    repository root.
    """
    base = responses(**overrides)
    base["rev-parse --show-prefix"] = "sub/\n"

    return base


@pytest.mark.asyncio
async def test_diffs_are_scoped_and_reported_relative_to_the_executor_directory(
    metadata,
):
    """`--numstat` paths are fed straight back as pathspecs, which are read relative to the working directory, so the
    two have to be in the same coordinate system."""
    fake = FakeGit(
        nested(
            **{
                f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat("1\t1\ta.rb"),
                f"{DIFF} {BASE_SHA} HEAD": (
                    "diff --git a/a.rb b/a.rb\n"
                    "--- a/a.rb\n+++ b/a.rb\n"
                    "@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n"
                ),
            }
        )
    )

    result = await run_tool(fake, metadata, mode="branch")

    assert '<file_diff filename="a.rb">' in result
    # The pathspec must be what numstat reported, not a root-relative rewrite of it.
    content = next(
        argv
        for argv in fake.git_argvs
        if argv[0] == "diff" and "--numstat" not in argv and "--" in argv
    )
    assert content[content.index("--") + 1 :] == ["a.rb"]


@pytest.mark.asyncio
async def test_diff_is_always_relative_to_the_executor_directory(metadata):
    """Without `--relative` a diff run below the root reports root-relative paths, which then match nothing when fed
    back as pathspecs, and the review silently covers no files."""
    assert "--relative" in DIFF_FLAGS

    fake = FakeGit(nested())
    await run_tool(fake, metadata, mode="branch")

    for argv in fake.git_argvs:
        if argv[0] == "diff":
            assert "--relative" in argv, argv


@pytest.mark.asyncio
async def test_status_paths_are_rebased_onto_the_executor_directory(metadata):
    """`status --porcelain` has no `--relative`, so its paths arrive root-relative."""
    fake = FakeGit(
        nested(status=numstat(" M sub/other.rb", "?? sub/scratch.txt")),
        files={"scratch.txt": "new\n"},
    )

    result = await run_tool(fake, metadata, mode="branch")

    assert "- other.rb" in result
    assert "- sub/other.rb" not in result


@pytest.mark.asyncio
async def test_changes_outside_the_executor_directory_are_left_out(metadata):
    """The executor refuses any read resolving outside the workspace folder, so a file above it could be named in the
    coverage contract but never opened."""
    fake = FakeGit(
        nested(status=numstat(" M top.rb", " M sub/inside.rb", "?? elsewhere/new.rb"))
    )

    result = await run_tool(fake, metadata, mode="branch")

    assert "- inside.rb" in result
    assert "top.rb" not in result
    assert "elsewhere/new.rb" not in result


@pytest.mark.asyncio
async def test_the_narrowed_scope_is_stated_in_the_payload(metadata):
    """Silently reviewing a subset reads to the user as "nothing else changed"."""
    fake = FakeGit(nested())

    result = await run_tool(fake, metadata, mode="branch")

    assert "<scope>" in result
    assert "sub/" in result


@pytest.mark.asyncio
async def test_no_scope_note_when_running_at_the_repository_root(metadata):
    fake = FakeGit(responses())

    result = await run_tool(fake, metadata, mode="branch")

    assert "<scope>" not in result


@pytest.mark.asyncio
async def test_untracked_files_below_the_root_are_read_where_the_executor_looks(
    metadata,
):
    """`read_file` resolves against the workspace folder, so the path has to be relative to it."""
    fake = FakeGit(
        nested(status=numstat("?? sub/scratch.txt")),
        files={"scratch.txt": "new line\n"},
    )

    result = await run_tool(fake, metadata, mode="working_tree")

    assert '<file_diff filename="scratch.txt">' in result
    reads = [
        a.runReadFile.filepath
        for a in fake.actions
        if a.WhichOneof("action") == "runReadFile"
    ]
    assert "scratch.txt" in reads
    assert "sub/scratch.txt" not in reads


@pytest.mark.asyncio
async def test_a_rename_outside_the_scope_still_consumes_its_origin_field(metadata):
    """The origin field has to be consumed even when the entry is dropped, or every later entry is parsed one field out
    of step."""
    fake = FakeGit(
        nested(status=numstat("R  top/new.rb", "top/old.rb", " M sub/after.rb"))
    )

    result = await run_tool(fake, metadata, mode="branch")

    assert "- after.rb" in result
    assert "old.rb" not in result
