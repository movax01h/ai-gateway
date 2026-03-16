"""Work Items Tests.

Validates posting data to issues, epics, and merge requests.
"""

import pytest

from .helpers import SAMPLE_ISSUES, glql_response, mock_glql_response
from agent_tests.helpers import ask_agent


@pytest.mark.asyncio
async def test_post_summary_to_issue(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """When user specifies 'post a summary', agent should query data then attempt to post."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES, count=42))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "How many open issues are in gitlab-org group? Post a summary to issue #1 in gitlab-org/gitlab-test",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    result.assert_has_tool_calls().assert_called_tool("create_work_item_note")
