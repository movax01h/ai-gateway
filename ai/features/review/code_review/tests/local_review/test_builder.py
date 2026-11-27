"""Payload assembly end to end, including the instructions-only mode."""

import re

import pytest
from langchain_core.tools import ToolException

from ai.features.review.code_review.components.local_review.exclusions import (
    MAX_DIFF_LINES_PER_FILE,
)
from ai.features.review.code_review.tests.conftest import (
    BASE_SHA,
    DIFF,
    DIFF_ARGV,
    FakeGit,
    numstat,
    responses,
    run_tool,
)

TRUNCATED = "\n\n[... output truncated: showing first and last 25KB of 80.0KB ...]\n\n"

RUBY_INSTRUCTIONS = """
instructions:
    -   name: Ruby Style
        fileFilters: ["**/*.rb"]
        instructions: Follow Ruby naming conventions
"""


@pytest.mark.asyncio
async def test_empty_diff_fails_instead_of_reviewing_nothing(metadata):
    """`execution_result` is the only thing a router can branch on, so an empty diff has to fail.

    A returned "nothing changed" payload would be handed to the reviewer to review.
    """
    fake = FakeGit(responses(**{f"{DIFF} --numstat -z {BASE_SHA} HEAD": ""}))

    with pytest.raises(ToolException) as raised:
        await run_tool(fake, metadata, mode="branch")

    # The message has to say which comparison came up empty; "nothing to review" alone
    # leaves the user unable to tell a clean branch from a wrong base ref.
    assert "nothing to review" in str(raised.value)
    assert "origin/main" in str(raised.value)
    assert not any(argv[0] == "diff" and "--" in argv for argv in fake.git_argvs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "served",
    [
        {"status": f"?? a.py\0{TRUNCATED}?? b.py\0"},
        {
            f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat("1\t1\tapp/a.rb")
            + TRUNCATED
            + numstat("1\t1\tapp/z.rb")
        },
    ],
)
async def test_a_truncated_listing_fails_rather_than_dropping_files(metadata, served):
    """The executor keeps the head and tail of a long listing; the files in the cut middle would vanish unnamed."""
    fake = FakeGit(responses(**served))

    with pytest.raises(ToolException, match="too large for the executor"):
        await run_tool(fake, metadata, mode="branch")


@pytest.mark.asyncio
async def test_only_instructions_with_no_changed_files_is_explicit(metadata):
    """An empty tool response would reach the reviewer prompt as a blank instruction slot."""
    fake = FakeGit(responses(**{f"{DIFF} --numstat -z {BASE_SHA} HEAD": ""}))

    result = await run_tool(fake, metadata, only_instructions=True)

    assert result == "<empty_diff>true</empty_diff>"


@pytest.mark.asyncio
async def test_only_instructions_returns_paths_and_instructions(metadata):
    instructions = RUBY_INSTRUCTIONS
    fake = FakeGit(
        responses(), files={".gitlab/duo/mr-review-instructions.yaml": instructions}
    )

    result = await run_tool(fake, metadata, only_instructions=True)

    assert "<changed_files>\n- app/a.rb\n</changed_files>" in result
    assert "<custom_instructions>" in result
    assert "Follow Ruby naming conventions" in result
    assert "<git_diffs>" not in result
    # No diff content is fetched in this mode.
    assert not any(
        argv[: len(DIFF_ARGV)] == DIFF_ARGV and "--numstat" not in argv
        for argv in fake.git_argvs
    )


@pytest.mark.asyncio
async def test_missing_instructions_file_is_not_an_error(metadata):
    fake = FakeGit(responses())

    result = await run_tool(fake, metadata, only_instructions=True)

    assert "<changed_files>" in result
    assert "<custom_instructions>" not in result


@pytest.mark.asyncio
async def test_only_instructions_scopes_untracked_files_to_the_mode(metadata):
    """Untracked files are out of scope in branch mode, so they must not drive instruction matching."""
    fake = FakeGit(
        responses(
            status=numstat("?? untracked.rb"),
            **{
                f"{DIFF} --numstat -z {BASE_SHA}": numstat("1\t1\tapp/a.rb"),
            },
        )
    )

    branch = await run_tool(fake, metadata, only_instructions=True, mode="branch")
    working_tree = await run_tool(
        fake, metadata, only_instructions=True, mode="working_tree"
    )

    assert "- untracked.rb" not in branch
    assert "- untracked.rb" in working_tree


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry", "path"),
    [
        ("-\t-\tblob.bin", "blob.bin"),
        (f"{MAX_DIFF_LINES_PER_FILE + 1}\t0\thuge.rb", "huge.rb"),
        ("1\t1\tdebug.log", "debug.log"),
        ("2\t0\t.env", ".env"),
    ],
    ids=["binary", "oversized", "policy", "denylist"],
)
async def test_lightweight_mode_reports_only_reviewable_paths(metadata, entry, path):
    """A path lightweight mode lists must be one the full review actually reviews.

    Otherwise a custom instruction matching only that path is activated for a file the reviewer never sees.
    """
    fake = FakeGit(
        responses(
            **{f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat(entry, "1\t1\tapp/a.rb")}
        )
    )

    result = await run_tool(fake, metadata, only_instructions=True)

    assert "- app/a.rb" in result
    assert f"- {path}" not in result


@pytest.mark.asyncio
async def test_lightweight_and_full_modes_agree_on_selection(metadata):
    """The two modes share one selection path, so their views cannot drift apart."""
    entries = numstat(
        "-\t-\tblob.bin",
        f"{MAX_DIFF_LINES_PER_FILE + 1}\t0\thuge.rb",
        "1\t1\tdebug.log",
        "2\t0\t.env",
        "1\t1\tapp/a.rb",
    )
    fake = FakeGit(responses(**{f"{DIFF} --numstat -z {BASE_SHA} HEAD": entries}))

    full = await run_tool(fake, metadata)
    light = await run_tool(fake, metadata, only_instructions=True)

    reviewed = set(re.findall(r'<file_diff filename="([^"]+)">', full))
    listed = set(re.findall(r"^- (\S+)$", light, re.MULTILINE))

    assert reviewed == listed == {"app/a.rb"}


@pytest.mark.asyncio
async def test_default_payload_carries_the_custom_instructions(metadata):
    """A deterministic step calls the tool once, so instructions have to arrive with the diff."""
    fake = FakeGit(
        responses(),
        files={".gitlab/duo/mr-review-instructions.yaml": RUBY_INSTRUCTIONS},
    )

    result = await run_tool(fake, metadata)

    assert "<custom_instructions>" in result
    assert "Follow Ruby naming conventions" in result
    assert "<git_diffs>" in result
    # Ambient renders the instructions before the coverage list; local matches so the
    # payloads can be compared without accounting for ordering.
    assert result.index("<custom_instructions>") < result.index("<changed_files>")
    assert result.index("<changed_files>") < result.index("<git_diffs>")


@pytest.mark.asyncio
async def test_payload_never_carries_the_attribution_hint(metadata):
    """The terminal step renders attribution from `custom_instruction_ref`.

    Sending the hint as well would make the model write the prefix too, attributing every instruction-driven comment
    twice.
    """
    fake = FakeGit(
        responses(),
        files={".gitlab/duo/mr-review-instructions.yaml": RUBY_INSTRUCTIONS},
    )

    for kwargs in ({}, {"only_instructions": True}):
        result = await run_tool(fake, metadata, **kwargs)

        assert "<custom_instructions>" in result, kwargs
        assert "According to custom instructions in" not in result, kwargs


@pytest.mark.asyncio
async def test_instructions_are_not_read_when_every_file_was_excluded(metadata):
    """Nothing is under review, so there is nothing for an instruction to match."""
    fake = FakeGit(
        responses(**{f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat("2\t0\t.env")}),
        files={".gitlab/duo/mr-review-instructions.yaml": RUBY_INSTRUCTIONS},
    )

    result = await run_tool(fake, metadata, mode="branch")

    assert "<custom_instructions>" not in result
    assert not any(
        a.WhichOneof("action") == "runReadFile"
        and a.runReadFile.filepath.endswith("mr-review-instructions.yaml")
        for a in fake.actions
    )


@pytest.mark.asyncio
async def test_only_instructions_distinguishes_empty_from_all_excluded(metadata):
    """The full path is careful about this distinction; the instructions path has to match it.

    Reporting an empty diff when files did change reads as "nothing to review".
    """
    fake = FakeGit(
        responses(**{f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat("2\t0\t.env")})
    )

    result = await run_tool(fake, metadata, only_instructions=True, mode="branch")

    assert "<all_files_excluded>" in result
    assert "<empty_diff>" not in result


@pytest.mark.asyncio
async def test_only_instructions_still_reports_a_genuinely_empty_diff(metadata):
    fake = FakeGit(responses(**{f"{DIFF} --numstat -z {BASE_SHA} HEAD": ""}))

    result = await run_tool(fake, metadata, only_instructions=True, mode="branch")

    assert result == "<empty_diff>true</empty_diff>"
