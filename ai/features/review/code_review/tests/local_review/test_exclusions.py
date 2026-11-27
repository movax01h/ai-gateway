"""Which paths are reviewable, and the reason recorded for every one that is not."""

from unittest.mock import patch

import pytest
from langchain_core.tools import ToolException

from ai.features.review.code_review.components.local_review import (
    LocalReviewContextBuilder,
)
from ai.features.review.code_review.components.local_review.exclusions import (
    CHECK_ATTR_BATCH,
    MAX_DIFF_LINES_PER_FILE,
    MAX_UNTRACKED_FILES,
)
from ai.features.review.code_review.tests.conftest import (
    APP_DIFF,
    BASE_SHA,
    DIFF,
    DIFF_ARGV,
    FakeGit,
    _strip_global_flags,
    numstat,
    responses,
    run_tool,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        ("-\t-\tblob.bin", "binary file"),
        (f"{MAX_DIFF_LINES_PER_FILE + 1}\t0\thuge.rb", "changed lines"),
        ("1\t1\tdebug.log", "project policy"),
    ],
    ids=["binary", "oversized", "policy"],
)
async def test_files_are_excluded_not_truncated(metadata, entry, reason):
    """Mode-agnostic, but pinned to branch so the asserted argv stays literal."""
    fake = FakeGit(
        responses(
            **{f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat(entry, "1\t1\tapp/a.rb")}
        )
    )

    result = await run_tool(fake, metadata, mode="branch")

    path = entry.split("\t")[-1]
    assert f"- {path}: " in result
    assert reason in result
    assert f'<file_diff filename="{path}">' not in result
    assert '<file_diff filename="app/a.rb">' in result
    # The excluded path is never sent to git, so no content is fetched for it.
    content_argvs = [
        argv for argv in fake.git_argvs if argv[0] == "diff" and "--" in argv
    ]
    assert content_argvs == [[*DIFF_ARGV, BASE_SHA, "HEAD", "--", "app/a.rb"]]


@pytest.mark.asyncio
async def test_untracked_files_are_included_only_in_working_tree_mode(metadata):
    fake = FakeGit(
        responses(
            status=numstat("?? new_file.rb"),
            **{
                f"{DIFF} --numstat -z {BASE_SHA}": numstat("1\t1\tapp/a.rb"),
                f"{DIFF} {BASE_SHA}": APP_DIFF,
            },
        ),
        files={"new_file.rb": "line one\nline two\n"},
    )

    result = await run_tool(fake, metadata, mode="working_tree")

    assert '<file_diff filename="new_file.rb">' in result
    assert '<line type="added" old_line="" new_line="1">line one</line>' in result
    assert '<line type="added" old_line="" new_line="2">line two</line>' in result
    assert "<chunk_header>@@ -0,0 +1,2 @@</chunk_header>" in result


@pytest.mark.asyncio
async def test_binary_untracked_file_is_excluded(metadata):
    fake = FakeGit(
        responses(
            status=numstat("?? blob.bin"),
            **{
                f"{DIFF} --numstat -z {BASE_SHA}": numstat("1\t1\tapp/a.rb"),
                f"{DIFF} {BASE_SHA}": APP_DIFF,
            },
        ),
        files={"blob.bin": "abc\0def"},
    )

    result = await run_tool(fake, metadata, mode="working_tree")

    assert "- blob.bin: binary or non-UTF-8 content" in result
    assert '<file_diff filename="blob.bin">' not in result


@pytest.mark.asyncio
async def test_an_untracked_file_the_executor_cannot_read_gets_its_own_reason(metadata):
    """A refused or failed read is not binary content, and the reviewer is told which it was."""
    fake = FakeGit(
        responses(
            status=numstat("?? gone.txt"),
            **{
                f"{DIFF} --numstat -z {BASE_SHA}": numstat("1\t1\tapp/a.rb"),
                f"{DIFF} {BASE_SHA}": APP_DIFF,
            },
        ),
        files={},
    )

    result = await run_tool(fake, metadata, mode="working_tree")

    assert "- gone.txt: could not be read from the checkout" in result
    assert "non-UTF-8" not in result
    assert '<file_diff filename="gone.txt">' not in result


@pytest.mark.asyncio
async def test_untracked_files_are_capped(metadata):
    count = MAX_UNTRACKED_FILES + 3
    paths = [f"scratch{i:02d}.txt" for i in range(count)]
    fake = FakeGit(
        responses(
            status=numstat(*(f"?? {p}" for p in paths)),
            **{
                f"{DIFF} --numstat -z {BASE_SHA}": numstat("1\t1\tapp/a.rb"),
                f"{DIFF} {BASE_SHA}": APP_DIFF,
            },
        ),
        files=dict.fromkeys(paths, "x\n"),
    )

    result = await run_tool(fake, metadata, mode="working_tree")

    assert result.count('<file_diff filename="scratch') == MAX_UNTRACKED_FILES
    assert f"more than {MAX_UNTRACKED_FILES} untracked files" in result
    # The cap lands before `check-attr`, so the command line never carries the whole list.
    attr_argv = next(argv for argv in fake.git_argvs if argv[0] == "check-attr")
    assert sum(a.startswith("scratch") for a in attr_argv) == MAX_UNTRACKED_FILES


@pytest.mark.asyncio
async def test_generated_files_are_excluded(metadata):
    """Parity with ambient review, which never shows the reviewer a generated file."""
    fake = FakeGit(
        responses(
            **{
                f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat(
                    "20\t0\tcontract_pb2.py", "1\t1\tapp/a.rb"
                ),
                "check-attr": "\0".join(
                    [
                        "contract_pb2.py",
                        "gitlab-generated",
                        "set",
                        "contract_pb2.py",
                        "linguist-generated",
                        "unspecified",
                        "app/a.rb",
                        "gitlab-generated",
                        "unspecified",
                        "app/a.rb",
                        "linguist-generated",
                        "unspecified",
                    ]
                )
                + "\0",
            }
        )
    )

    result = await run_tool(fake, metadata)

    assert "- contract_pb2.py: generated file" in result
    assert '<file_diff filename="contract_pb2.py">' not in result
    assert '<file_diff filename="app/a.rb">' in result
    # The generated path is never sent to git for content.
    content_argv = next(
        argv for argv in fake.git_argvs if argv[0] == "diff" and "--" in argv
    )
    assert "contract_pb2.py" not in content_argv


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "excluded"),
    [("set", True), ("unset", False), ("unspecified", False)],
)
async def test_only_an_explicit_set_attribute_means_generated(
    metadata, value, excluded
):
    """`unset` is a deliberate "not generated" override; `unspecified` means nothing declared."""
    fake = FakeGit(
        responses(
            **{"check-attr": "\0".join(["app/a.rb", "gitlab-generated", value]) + "\0"}
        )
    )

    result = await run_tool(fake, metadata)

    assert ("- app/a.rb: generated file" in result) is excluded
    assert ('<file_diff filename="app/a.rb">' in result) is not excluded


@pytest.mark.asyncio
async def test_generated_untracked_files_are_excluded_in_working_tree_mode(metadata):
    fake = FakeGit(
        responses(
            status=numstat("?? build/out_pb2.py"),
            **{
                f"{DIFF} --numstat -z {BASE_SHA}": numstat("1\t1\tapp/a.rb"),
                f"{DIFF} {BASE_SHA}": APP_DIFF,
                "check-attr": "\0".join(["build/out_pb2.py", "gitlab-generated", "set"])
                + "\0",
            },
        ),
        files={"build/out_pb2.py": "generated\n"},
    )

    result = await run_tool(fake, metadata, mode="working_tree")

    assert "- build/out_pb2.py: generated file" in result
    assert '<file_diff filename="build/out_pb2.py">' not in result


@pytest.mark.asyncio
async def test_generated_files_do_not_drive_instruction_matching(metadata):
    """Ambient review filters generated files before the paths reach its lightweight mode."""
    fake = FakeGit(
        responses(
            **{
                f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat(
                    "20\t0\tgen_pb2.py", "1\t1\tapp/a.rb"
                ),
                "check-attr": "\0".join(["gen_pb2.py", "gitlab-generated", "set"])
                + "\0",
            }
        )
    )

    result = await run_tool(fake, metadata, only_instructions=True)

    assert "- app/a.rb" in result
    assert "gen_pb2.py" not in result


@pytest.mark.asyncio
async def test_no_check_attr_call_when_nothing_changed(metadata):
    fake = FakeGit(responses(**{f"{DIFF} --numstat -z {BASE_SHA} HEAD": ""}))

    await run_tool(fake, metadata, only_instructions=True)

    assert not any(argv[0] == "check-attr" for argv in fake.git_argvs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.production",
        ".ssh/id_rsa",
        "Dockerfile.secrets",
        ".gitlab/duo/rules.yaml",
    ],
)
async def test_denylisted_paths_never_reach_the_diff(metadata, path):
    """The tool is preapproved under `read_only_files`, so it must honour the same denylist `read_file` does; otherwise
    a changed `.env` is returned with no approval."""
    fake = FakeGit(
        responses(
            **{
                f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat(
                    f"2\t0\t{path}", "1\t1\tapp/a.rb"
                )
            }
        )
    )

    result = await run_tool(fake, metadata)

    assert f"- {path}: excluded by Duo context exclusions" in result
    assert f'<file_diff filename="{path}">' not in result
    # The path is never handed to git, so its content is never fetched.
    content_argv = next(
        argv for argv in fake.git_argvs if argv[0] == "diff" and "--" in argv
    )
    assert path not in content_argv


@pytest.mark.asyncio
async def test_denylisted_old_path_excludes_the_rename(metadata):
    """A rename out of a denylisted directory would otherwise put the old path in the pathspec."""
    fake = FakeGit(
        responses(
            **{
                f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat(
                    "0\t0\t", ".ssh/id_rsa", "docs/key.txt"
                )
            }
        )
    )

    result = await run_tool(fake, metadata)

    assert "- docs/key.txt: excluded by Duo context exclusions" in result
    assert ".ssh/id_rsa" not in [a for argv in fake.git_argvs for a in argv]


@pytest.mark.asyncio
async def test_denylisted_untracked_content_is_never_read(metadata):
    fake = FakeGit(
        responses(
            status=numstat("?? .env.staging"),
            **{
                f"{DIFF} --numstat -z {BASE_SHA}": numstat("1\t1\tapp/a.rb"),
                f"{DIFF} {BASE_SHA}": APP_DIFF,
            },
        ),
        files={".env.staging": "UNTRACKED_SECRET=leaked\n"},
    )

    result = await run_tool(fake, metadata, mode="working_tree")

    assert "- .env.staging: excluded by Duo context exclusions" in result
    assert "UNTRACKED_SECRET" not in result
    assert not any(
        a.WhichOneof("action") == "runReadFile"
        and a.runReadFile.filepath == ".env.staging"
        for a in fake.actions
    )


@pytest.mark.asyncio
async def test_instructions_file_is_readable_but_not_reviewable(metadata):
    """`.gitlab/duo` is denylisted, so the instructions file is read for its declared purpose only and is still kept out
    of the reviewed diff."""
    instructions = (
        'instructions:\n  - name: Style\n    fileFilters: ["*.rb"]\n'
        "    instructions: Be nice\n"
    )
    fake = FakeGit(
        responses(
            **{
                f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat(
                    "1\t1\tapp/a.rb", "2\t0\t.gitlab/duo/mr-review-instructions.yaml"
                )
            }
        ),
        files={".gitlab/duo/mr-review-instructions.yaml": instructions},
    )

    review = await run_tool(fake, metadata)
    assert (
        "- .gitlab/duo/mr-review-instructions.yaml: excluded by Duo context exclusions"
        in review
    )
    assert 'filename=".gitlab/duo/mr-review-instructions.yaml"' not in review

    fake.actions.clear()
    instructions_out = await run_tool(fake, metadata, only_instructions=True)
    assert "Be nice" in instructions_out


@pytest.mark.asyncio
async def test_falls_back_when_check_attr_source_is_unsupported(metadata):
    """`--source` needs git >= 2.40; an older git must degrade rather than fail the review."""

    class OldGit(FakeGit):
        def __call__(self, metadata, action):
            if action.WhichOneof("action") == "runCommand":
                argv = list(action.runCommand.arguments)
                if any(a.startswith("--source") for a in argv):
                    self.actions.append(action)
                    raise ToolException("Action error: unknown option `source'")
            return super().__call__(metadata, action)

    fake = OldGit(responses())

    result = await run_tool(fake, metadata, mode="branch")

    attr_argvs = [argv for argv in fake.git_argvs if argv[0] == "check-attr"]
    assert any(a.startswith("--source") for a in attr_argvs[0])
    assert not any(a.startswith("--source") for a in attr_argvs[1])
    assert '<file_diff filename="app/a.rb">' in result


@pytest.mark.asyncio
async def test_untracked_file_over_the_line_cap_is_excluded(metadata):
    """The cap applies to new files too; an untracked blob has no numstat to catch it earlier."""
    body = "\n".join(f"line {n}" for n in range(MAX_DIFF_LINES_PER_FILE + 1)) + "\n"
    fake = FakeGit(
        responses(status=numstat("?? huge.txt")),
        files={"huge.txt": body},
    )

    result = await run_tool(fake, metadata, mode="working_tree")

    assert f"huge.txt: more than {MAX_DIFF_LINES_PER_FILE} changed lines" in result
    assert '<file_diff filename="huge.txt">' not in result


@pytest.mark.asyncio
async def test_policy_excluded_untracked_file_is_never_read(metadata):
    """`exclusion_rules` covers untracked paths too, and the content must not be fetched."""
    fake = FakeGit(
        responses(status=numstat("?? debug.log")),
        files={"debug.log": "secret\n"},
    )

    result = await run_tool(fake, metadata, mode="working_tree")

    assert "debug.log: excluded by project policy" in result
    assert not any(
        a.WhichOneof("action") == "runReadFile"
        and a.runReadFile.filepath == "debug.log"
        for a in fake.actions
    )


@pytest.mark.asyncio
async def test_a_truncated_check_attr_listing_fails_rather_than_keeping_files(metadata):
    """A generated or filter-managed file in the cut middle would be reviewed as if it were neither."""
    fake = FakeGit(
        responses(
            **{
                "check-attr": "app/a.rb\0gitlab-generated\0unspecified\0"
                "\n\n[... output truncated: showing first and last 25KB of 80.0KB ...]\n\n"
                "app/a.rb\0filter\0unspecified\0"
            }
        )
    )

    with pytest.raises(ToolException, match="too large for the executor"):
        await run_tool(fake, metadata, mode="branch")


@pytest.mark.asyncio
async def test_check_attr_paths_are_sent_in_batches(metadata):
    """Three triples per path add up: one command for a few hundred paths would overrun the output cap."""
    paths = [f"app/f{index:03}.rb" for index in range(CHECK_ATTR_BATCH * 2 + 1)]
    fake = FakeGit(
        responses(
            **{
                f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat(
                    *(f"1\t1\t{path}" for path in paths)
                )
            }
        )
    )

    await run_tool(fake, metadata, mode="branch", only_instructions=True)

    batches = [
        argv[argv.index("--") + 1 :]
        for argv in fake.git_argvs
        if argv[0] == "check-attr"
    ]
    assert [len(batch) for batch in batches] == [CHECK_ATTR_BATCH, CHECK_ATTR_BATCH, 1]
    assert sorted(path for batch in batches for path in batch) == paths


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["working_tree", "branch"])
async def test_check_attr_failure_is_not_swallowed(metadata, mode):
    """Only an old git rejecting `--source` is tolerated; any other check-attr failure is a real failure.

    `working_tree` sends no `--source`, so there is no fallback to hide behind. `branch` sends one, and the failure
    must still surface rather than silently re-reading attributes from the working tree.
    """
    fake = FakeGit(responses(status=numstat("?? new.py")))

    def failing(_metadata, action):
        argv = _strip_global_flags(list(action.runCommand.arguments))
        if argv[0] == "check-attr":
            raise ToolException("Action error: check-attr exploded")
        return FakeGit.__call__(fake, _metadata, action)

    tool = LocalReviewContextBuilder(metadata, metadata["project"])
    with patch(
        "ai.features.review.code_review.components.local_review.transport._execute_action",
        side_effect=failing,
    ):
        with pytest.raises(ToolException, match="check-attr exploded"):
            await tool.build(mode=mode)


@pytest.mark.asyncio
async def test_filtered_paths_are_excluded_from_the_review(metadata):
    """An LFS-tracked file would diff as pointer against content once the driver is emptied.

    Excluding it beats reporting a diff that never existed.
    """
    fake = FakeGit(
        responses(
            **{
                f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat(
                    "1\t1\tapp/a.rb", "2\t2\tbig.psd"
                ),
                "check-attr": "\0".join(
                    ["big.psd", "filter", "lfs", "app/a.rb", "filter", "unspecified"]
                )
                + "\0",
            }
        )
    )

    result = await run_tool(fake, metadata, mode="branch")

    assert "big.psd: managed by a git filter" in result
    assert '<file_diff filename="big.psd">' not in result
    assert "- app/a.rb" in result


@pytest.mark.asyncio
async def test_a_rename_out_of_a_filtered_path_is_still_reviewed(metadata):
    """Only the new path decides, because only the worktree side is affected by emptying a driver.

    The old side is a stored blob either way, so the diff a rename out of a filtered directory produces is the same one
    plain git produces with the driver active. It reports a real change -- an LFS object committed as raw content, say
    -- and hiding it would hide the mistake.
    """
    rename_diff = """diff --git a/vendor/data.txt b/src/data.txt
similarity index 90%
rename from vendor/data.txt
rename to src/data.txt
--- a/vendor/data.txt
+++ b/src/data.txt
@@ -1,1 +1,1 @@
-token = REDACTED
+token = SECRET
"""
    fake = FakeGit(
        responses(
            **{
                f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat(
                    "1\t1\t", "vendor/data.txt", "src/data.txt"
                ),
                f"{DIFF} {BASE_SHA} HEAD": rename_diff,
                "check-attr": "\0".join(["src/data.txt", "filter", "unspecified"])
                + "\0",
            }
        )
    )

    result = await run_tool(fake, metadata, mode="branch")

    assert '<file_diff filename="src/data.txt">' in result
    assert "managed by a git filter" not in result
    # The old path is not queried: it has no worktree content for a driver to rewrite.
    attr_argv = next(argv for argv in fake.git_argvs if argv[0] == "check-attr")
    assert "vendor/data.txt" not in attr_argv


@pytest.mark.asyncio
async def test_an_unset_filter_attribute_does_not_exclude(metadata):
    """`unset` and `unspecified` both mean no driver rewrites the file."""
    fake = FakeGit(
        responses(
            **{
                "check-attr": "\0".join(["app/a.rb", "filter", "unset"]) + "\0",
            }
        )
    )

    result = await run_tool(fake, metadata, mode="branch")

    assert '<file_diff filename="app/a.rb">' in result
    assert "managed by a git filter" not in result


@pytest.mark.asyncio
async def test_filtered_untracked_file_is_excluded_and_never_read(metadata):
    fake = FakeGit(
        responses(
            status=numstat("?? blob.bin"),
            **{"check-attr": "\0".join(["blob.bin", "filter", "lfs"]) + "\0"},
        ),
        files={"blob.bin": "pointer\n"},
    )

    result = await run_tool(fake, metadata, mode="working_tree")

    assert "blob.bin: managed by a git filter" in result
    assert not any(
        a.WhichOneof("action") == "runReadFile" and a.runReadFile.filepath == "blob.bin"
        for a in fake.actions
    )
