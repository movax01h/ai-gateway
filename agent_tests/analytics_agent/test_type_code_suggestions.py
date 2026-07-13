"""Code Suggestions data source tests.

Validates that the agent generates correct GLQL queries for Code Suggestions analytics, including analytics mode,
dimensions, metrics, filters, and result interpretation.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import (
    SAMPLE_CODE_SUGGESTIONS,
    SAMPLE_CODE_SUGGESTIONS_BY_IDE,
    glql_analytics_response,
    mock_glql_response,
)


@pytest.mark.asyncio
async def test_acceptance_rate_by_language(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
) -> None:
    """Agent should generate analytics-mode query for acceptance rate by language."""
    mock_glql_response(
        mock_gitlab_client, glql_analytics_response(SAMPLE_CODE_SUGGESTIONS)
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "What's the acceptance rate for code suggestions by language?",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses mode: analytics",
            "The GLQL query includes type = CodeSuggestion",
            "The GLQL query uses language as a dimension",
            "The GLQL query includes acceptanceRate as a metric",
        ]
    )


@pytest.mark.asyncio
async def test_suggestion_usage_by_ide(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
) -> None:
    """Agent should generate analytics-mode query for usage grouped by IDE."""
    mock_glql_response(
        mock_gitlab_client, glql_analytics_response(SAMPLE_CODE_SUGGESTIONS_BY_IDE)
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Which IDE has the highest code suggestion usage?",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses mode: analytics",
            "The GLQL query uses ideName as a dimension",
            "The GLQL query includes totalCount as a metric",
            "The response identifies an IDE with the highest usage",
        ]
    )


@pytest.mark.asyncio
async def test_does_not_use_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
) -> None:
    """Agent should use dimensions and metrics in analytics mode, not fields."""
    mock_glql_response(
        mock_gitlab_client, glql_analytics_response(SAMPLE_CODE_SUGGESTIONS)
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me code suggestion statistics by language",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses mode: analytics with dimensions and metrics, not fields",
        ]
    )
