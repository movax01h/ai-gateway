"""Post to GitLab Tests.

Validates posting analysis results to issues, epics, and merge requests.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import SAMPLE_ISSUES, SAMPLE_MRS, glql_response, mock_glql_response


@pytest.mark.asyncio
async def test_post_summary_to_issue(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Agent should query data then post to an issue."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES, count=42))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "How many open issues are in gitlab-org group? "
        "Post the full analysis as a comment on issue #1 "
        "in gitlab-org/gitlab-test",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    result.assert_has_tool_calls().assert_called_tool("create_work_item_note")


@pytest.mark.asyncio
async def test_post_summary_to_merge_request(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Agent should query data then post to a merge request."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "how many MR were merged this month in gitlab-org group? "
        "Post a short summary as a comment on "
        "merge request !1 in gitlab-org/gitlab-duo",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    result.assert_has_tool_calls().assert_called_tool("create_merge_request_note")
