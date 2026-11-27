"""What is being compared, and how the base ref resolves."""

from unittest.mock import patch

import pytest
from langchain_core.tools import ToolException

from ai.features.review.code_review.components.local_review import (
    LocalReviewContextBuilder,
)
from ai.features.review.code_review.components.local_review.transport import (
    GitCommandError,
)
from ai.features.review.code_review.tests.conftest import (
    APP_DIFF,
    BASE_SHA,
    DIFF,
    DIFF_ARGV,
    HEAD_SHA,
    FakeGit,
    _strip_global_flags,
    numstat,
    responses,
    run_tool,
)


@pytest.mark.asyncio
async def test_branch_mode_diffs_merge_base_against_head(metadata):
    fake = FakeGit(responses())

    result = await run_tool(fake, metadata, mode="branch")

    assert f"<base_sha>{BASE_SHA}</base_sha>" in result
    assert f"<head_sha>{HEAD_SHA}</head_sha>" in result
    assert "<mode>branch</mode>" in result
    assert "<base_ref>origin/main</base_ref>" in result
    assert '<file_diff filename="app/a.rb">' in result
    assert '<line type="added" old_line="" new_line="1">a = 2</line>' in result
    assert "<uncommitted_changes>" not in result
    assert "<empty_diff>" not in result
    assert [*DIFF_ARGV, BASE_SHA, "HEAD", "--", "app/a.rb"] in fake.git_argvs


@pytest.mark.asyncio
async def test_working_tree_mode_diffs_merge_base_against_the_working_tree(metadata):
    fake = FakeGit(
        responses(
            **{
                f"{DIFF} --numstat -z {BASE_SHA}": numstat("1\t1\tapp/a.rb"),
                f"{DIFF} {BASE_SHA}": APP_DIFF,
            }
        )
    )

    result = await run_tool(fake, metadata, mode="working_tree")

    assert "<mode>working_tree</mode>" in result
    assert [*DIFF_ARGV, BASE_SHA, "--", "app/a.rb"] in fake.git_argvs
    assert not any(
        "HEAD" in argv[len(DIFF_ARGV) :] for argv in fake.git_argvs if argv[0] == "diff"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_ref",
    ["--upload-pack=/bin/sh", "-1", "main;rm -rf /", "origin/main..HEAD", "a b"],
)
async def test_invalid_base_ref_is_rejected(metadata, base_ref):
    fake = FakeGit(responses())

    with pytest.raises(ToolException, match="not a valid git ref name"):
        await run_tool(fake, metadata, base_ref=base_ref)

    assert fake.actions == []


@pytest.mark.asyncio
async def test_unknown_base_ref_is_rejected(metadata):
    fake = FakeGit(responses())

    with pytest.raises(ToolException, match="do not exist|none of"):
        await run_tool(fake, metadata, base_ref="origin/nope")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "match"),
    [
        (
            GitCommandError("rev-parse", "128", "fatal: not a git repository"),
            "exited with 128",
        ),
        (ToolException("Error: this checkout is a partial clone"), "partial clone"),
    ],
)
async def test_a_failure_other_than_a_missing_ref_is_not_read_as_missing(
    metadata, error, match
):
    """Only exit 1 means the ref is absent; telling the user to fetch the default branch would hide the real error."""
    fake = FakeGit(responses())

    def failing(_metadata, action):
        argv = _strip_global_flags(list(action.runCommand.arguments))
        if argv[:2] == ["rev-parse", "--verify"]:
            raise error
        return fake(_metadata, action)

    with patch(
        "ai.features.review.code_review.components.local_review.transport._execute_action",
        side_effect=failing,
    ):
        with pytest.raises(ToolException, match=match):
            await LocalReviewContextBuilder(metadata, metadata["project"]).build()


@pytest.mark.asyncio
async def test_base_ref_falls_back_to_the_local_default_branch(metadata):
    fake = FakeGit(responses(), known_refs={"main", "feature"})

    result = await run_tool(fake, metadata)

    assert "<base_ref>main</base_ref>" in result
    assert ["merge-base", "main", "HEAD"] in fake.git_argvs


@pytest.mark.asyncio
async def test_project_without_a_default_branch_fails_clearly(metadata):
    metadata = {**metadata, "project": {**metadata["project"], "default_branch": None}}
    fake = FakeGit(responses())

    with pytest.raises(ToolException, match="no base ref to compare against"):
        await run_tool(fake, metadata)


@pytest.mark.asyncio
async def test_a_tag_is_an_acceptable_base_ref(metadata):
    fake = FakeGit(responses(), known_refs={"main", "v1.0.0"})

    result = await run_tool(fake, metadata, base_ref="v1.0.0")

    assert "<base_ref>v1.0.0</base_ref>" in result
    # `^{commit}` is what makes an annotated tag resolve to the commit it points at.
    verify = next(
        argv for argv in fake.git_argvs if argv[:2] == ["rev-parse", "--verify"]
    )
    assert verify[-1] == "v1.0.0^{commit}"


@pytest.mark.asyncio
async def test_refs_are_verified_individually_not_enumerated(metadata):
    """Listing every ref overruns the output cap in a repository with a few thousand of them, and the truncated middle
    is where `origin/<default>` lives."""
    fake = FakeGit(responses())

    await run_tool(fake, metadata, mode="branch")

    assert not any(argv[0] == "for-each-ref" for argv in fake.git_argvs)
    assert any(argv[:2] == ["rev-parse", "--verify"] for argv in fake.git_argvs)


@pytest.mark.asyncio
async def test_the_remote_default_branch_is_preferred_over_the_local_one(metadata):
    """A stale local `main` would silently widen the review to everything since it diverged."""
    fake = FakeGit(responses(), known_refs={"main", "origin/main"})

    result = await run_tool(fake, metadata, mode="branch")

    assert "<base_ref>origin/main</base_ref>" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["branch", "working_tree"])
async def test_review_target_reports_whether_the_tree_is_dirty(metadata, mode):
    """Mode alone no longer implies mutability: a clean working_tree review is as stable as branch."""
    clean = FakeGit(responses())
    dirty = FakeGit(responses(status=numstat(" M app/a.rb")))

    assert (
        "<uncommitted_changes_present>false</uncommitted_changes_present>"
        in await run_tool(clean, metadata, mode=mode)
    )
    assert (
        "<uncommitted_changes_present>true</uncommitted_changes_present>"
        in await run_tool(dirty, metadata, mode=mode)
    )


@pytest.mark.asyncio
async def test_branch_mode_reads_attributes_from_the_reviewed_revision(metadata):
    """A dirty `.gitattributes` must not change which committed files are excluded.

    `check-attr` reads the working tree by default, so without `--source` an uncommitted attribute change would include
    or exclude committed files that are outside the stated review target.
    """
    fake = FakeGit(
        responses(
            **{
                f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat(
                    "1\t1\tapp/a.rb", "2\t0\tgen.pb.go"
                ),
                "check-attr": "\0".join(["gen.pb.go", "gitlab-generated", "set"])
                + "\0",
            }
        )
    )

    result = await run_tool(fake, metadata, mode="branch")

    attr_argv = next(argv for argv in fake.git_argvs if argv[0] == "check-attr")
    assert f"--source={HEAD_SHA}" in attr_argv
    assert "- gen.pb.go: generated file" in result


@pytest.mark.asyncio
async def test_working_tree_mode_reads_attributes_from_the_working_tree(metadata):
    """In working_tree mode the working tree *is* the reviewed revision, so no pinning."""
    fake = FakeGit(
        responses(
            **{
                f"{DIFF} --numstat -z {BASE_SHA}": numstat("1\t1\tapp/a.rb"),
                f"{DIFF} {BASE_SHA}": APP_DIFF,
            }
        )
    )

    await run_tool(fake, metadata, mode="working_tree")

    attr_argv = next(argv for argv in fake.git_argvs if argv[0] == "check-attr")
    assert not any(a.startswith("--source") for a in attr_argv)
