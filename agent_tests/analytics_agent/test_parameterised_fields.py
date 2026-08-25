"""Parameterised field tests.

Validates that the agent uses parameterised fields (e.g. ``durationQuantile``)
with aliases when the same field is queried with different parameters.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import (
    SAMPLE_PIPELINE_ANALYTICS_PERCENTILES,
    glql_analytics_response,
    mock_glql_response,
)


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_duration_percentiles_with_aliases(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
) -> None:
    """Agent should query duration quantiles with unique single-word aliases.

    The user question explicitly requests sorted output, so the agent is
    expected to sort by one of the aliases.
    """
    mock_glql_response(
        mock_gitlab_client,
        glql_analytics_response(SAMPLE_PIPELINE_ANALYTICS_PERCENTILES),
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Compare the median and p95 pipeline duration by ref for project "
        "gitlab-org/gitlab, sorted by median duration",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses mode: analytics",
            "The GLQL query includes type = Pipeline",
            "The GLQL query includes durationQuantile(0.5) as a metric",
            "The GLQL query includes durationQuantile(0.95) as a metric",
            "Each durationQuantile metric is assigned an alias using the `as` "
            "keyword. The alias text itself must contain no whitespace; "
            "surrounding double quotes are allowed and should be ignored when "
            'judging (e.g. `as p95`, `as "Median"`, and `as "medianDuration"` '
            'are all valid; `as "Median Duration"` is invalid because the '
            "alias text contains a space)",
            "The GLQL query contains a sort clause that references one of the "
            "aliases assigned to a durationQuantile metric",
            "The response compares the median and p95 pipeline durations",
        ]
    )
