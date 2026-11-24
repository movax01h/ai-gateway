# pylint: disable=file-naming-for-tests
"""The local payload and the merge-request payload must be the same product.

The reviewer prompt is shared: it reads line numbers out of `<line>` attributes and treats
`<changed_files>` as a coverage contract. If the two acquisition paths render either differently,
the same change reviewed locally and in a merge request gives different findings, and #2932's
comparison measures the formatter rather than the reviewer.

Both payloads are built through `BuildReviewMergeRequestContext`, with `surface` set each way, so
the comparison runs over the dispatch the flow actually uses rather than over two separate tools.
"""

import json
import re
from unittest.mock import AsyncMock, patch

import pytest

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.code_review.build_review_merge_request_context import (
    BuildReviewMergeRequestContext,
)
from tests.duo_workflow_service.tools.code_review.conftest import (
    BASE_SHA,
    DIFF,
    FakeGit,
    numstat,
    responses,
)

PATH = "calculator.rb"

# One hunk, exercising all three line types plus a chunk header.
HUNKS = (
    "@@ -4,7 +4,7 @@ class Calculator\n"
    "   end\n"
    " \n"
    "   def subtract(a, b)\n"
    "-    # TODO: Implement\n"
    "+    a - b\n"
    "   end\n"
    " end"
)

# The API hands over hunks; git hands over a file header as well. Same change either way.
GIT_DIFF = f"diff --git a/{PATH} b/{PATH}\nindex 111..222 100644\n--- a/{PATH}\n+++ b/{PATH}\n{HUNKS}\n"


def _section(payload: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", payload, re.DOTALL)
    assert match, f"<{tag}> missing from payload:\n{payload}"
    return match.group(1)


async def _merge_request_payload(metadata, gitlab_client_mock) -> str:
    """The ambient `fetch_mr_data` call: only_diffs, changed-files list, no format hint."""
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(
                status_code=200,
                body=json.dumps(
                    {
                        "id": 1,
                        "title": "Fix subtract",
                        "description": "It added instead of subtracting.",
                        "target_branch": "main",
                        "source_branch": "feature",
                    }
                ),
            ),
            GitLabHttpResponse(
                status_code=200,
                body=json.dumps(
                    [
                        {
                            "old_path": PATH,
                            "new_path": PATH,
                            "new_file": False,
                            "generated_file": False,
                            "renamed_file": False,
                            "diff": HUNKS,
                        }
                    ]
                ),
            ),
            # Custom instructions file: absent.
            GitLabHttpResponse(status_code=404, body="{}"),
        ]
    )

    return await BuildReviewMergeRequestContext(metadata=metadata).ainvoke(
        {
            "project_id": 1,
            "merge_request_iid": 1,
            "only_diffs": True,
            "include_changed_files_list": True,
            "include_instruction_format_hint": False,
        }
    )


async def _local_payload(metadata) -> str:
    """The same call with `surface: checkout`, replayed against a fake git."""
    fake = FakeGit(
        responses(
            **{
                f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat(f"1\t1\t{PATH}"),
                f"{DIFF} {BASE_SHA} HEAD": GIT_DIFF,
            }
        )
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)

    with patch(
        "duo_workflow_service.tools.code_review.local_review.transport._execute_action",
        side_effect=fake,
    ):
        return await tool.ainvoke(
            {
                "surface": "checkout",
                "mode": "branch",
                "only_diffs": True,
                "include_changed_files_list": True,
                "include_instruction_format_hint": False,
            }
        )


@pytest.mark.asyncio
async def test_both_paths_render_the_same_file_diff(metadata, gitlab_client_mock):
    """The `<file_diff>`/`<line>` representation is the reviewer's only source of line numbers."""
    ambient = await _merge_request_payload(metadata, gitlab_client_mock)
    local = await _local_payload(metadata)

    assert _section(local, "git_diffs") == _section(ambient, "git_diffs")


@pytest.mark.asyncio
async def test_both_paths_render_the_same_changed_files_list(
    metadata, gitlab_client_mock
):
    """The prompt treats this list as a coverage contract, so it has to mean the same thing."""
    ambient = await _merge_request_payload(metadata, gitlab_client_mock)
    local = await _local_payload(metadata)

    assert _section(local, "changed_files") == _section(ambient, "changed_files")
    assert _section(local, "changed_files") == f"- {PATH}"


@pytest.mark.asyncio
async def test_neither_path_preloads_file_contents(metadata, gitlab_client_mock):
    """Preloaded contents were ~68% of the payload, re-billed every cycle, with no recall gain."""
    ambient = await _merge_request_payload(metadata, gitlab_client_mock)
    local = await _local_payload(metadata)

    for payload in (ambient, local):
        assert "<original_files>" not in payload
        assert "<file_contents>" not in payload


@pytest.mark.asyncio
async def test_neither_path_sends_the_attribution_hint(metadata, gitlab_client_mock):
    ambient = await _merge_request_payload(metadata, gitlab_client_mock)
    local = await _local_payload(metadata)

    for payload in (ambient, local):
        assert "According to custom instructions in" not in payload


@pytest.mark.asyncio
async def test_local_omits_the_merge_request_only_tags(metadata):
    """An empty <mr_title> would tell the reviewer it has the author's intent when it has none."""
    local = await _local_payload(metadata)

    assert "<mr_title>" not in local
    assert "<mr_description>" not in local
    # What stands in for them instead.
    assert "<review_target>" in local
