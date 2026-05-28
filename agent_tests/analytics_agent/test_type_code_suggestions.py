"""Code Suggestions data source tests.

Validates that the agent generates correct GLQL queries for Code Suggestions analytics, including analytics mode,
dimensions, metrics, filters, and result interpretation.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import (
    SAMPLE_CODE_SUGGESTIONS,
    SAMPLE_CODE_SUGGESTIONS_BY_IDE,
    SAMPLE_CODE_SUGGESTIONS_BY_USER,
    glql_analytics_response,
    mock_glql_response,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sample_data, question, expected_tools, criteria",
    [
        pytest.param(
            SAMPLE_CODE_SUGGESTIONS,
            "What's the acceptance rate for code suggestions by language?",
            ["get_glql_schema", "run_glql_query"],
            [
                "The GLQL query uses mode: analytics",
                "The GLQL query includes type = CodeSuggestion",
                "The GLQL query uses language as a dimension",
                "The GLQL query includes acceptanceRate as a metric",
            ],
            id="acceptance_rate_by_language",
        ),
        pytest.param(
            SAMPLE_CODE_SUGGESTIONS_BY_IDE,
            "Which IDE has the highest code suggestion usage?",
            ["get_glql_schema", "run_glql_query"],
            [
                "The GLQL query uses mode: analytics",
                "The GLQL query uses ideName as a dimension",
                "The GLQL query includes totalCount as a metric",
                "The response identifies an IDE with the highest usage",
            ],
            id="suggestion_usage_by_ide",
        ),
        pytest.param(
            SAMPLE_CODE_SUGGESTIONS_BY_USER,
            "Who are the top users of code suggestions?",
            ["get_glql_schema", "run_glql_query"],
            [
                "The GLQL query uses mode: analytics",
                "The GLQL query uses user as a dimension",
                "The GLQL query includes totalCount as a metric",
            ],
            id="top_users",
        ),
        pytest.param(
            SAMPLE_CODE_SUGGESTIONS,
            "Show me code suggestion trends by language for the last 30 days",
            ["get_glql_schema", "run_glql_query"],
            [
                "The GLQL query uses mode: analytics",
                "The GLQL query includes type = CodeSuggestion",
                "The GLQL query filters by timestamp using a relative time expression",
            ],
            id="trends_by_language",
        ),
        pytest.param(
            [SAMPLE_CODE_SUGGESTIONS[0]],  # Python only
            "What's the acceptance rate for Python suggestions?",
            ["get_glql_schema", "run_glql_query"],
            [
                "The GLQL query uses mode: analytics",
                'The GLQL query filters by language = "python" or similar',
                "The GLQL query includes acceptanceRate as a metric",
                "The response mentions an acceptance rate percentage",
            ],
            id="python_acceptance_rate",
        ),
        pytest.param(
            SAMPLE_CODE_SUGGESTIONS,
            "What's the acceptance rate for code suggestions by language?",
            ["run_glql_query"],
            [
                "The response presents acceptance rates as percentages (e.g., 72%, 65%, 78%) rather than raw decimals",
            ],
            id="acceptance_rate_as_percentage",
        ),
        pytest.param(
            SAMPLE_CODE_SUGGESTIONS,
            "Show me code suggestion statistics by language",
            ["get_glql_schema", "run_glql_query"],
            [
                "The GLQL query uses mode: analytics with dimensions and metrics, not fields",
            ],
            id="does_not_use_fields",
        ),
        pytest.param(
            SAMPLE_CODE_SUGGESTIONS,
            "List all code suggestions",
            ["get_glql_schema", "run_glql_query"],
            [
                "The GLQL query uses mode: analytics, not standard mode",
                "The GLQL query does NOT use 'fields' parameter",
            ],
            id="does_not_use_standard_mode",
        ),
    ],
)
async def test_code_suggestions(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    sample_data,
    question,
    expected_tools,
    criteria,
):
    """Agent should generate correct analytics-mode GLQL queries for Code Suggestions."""
    mock_glql_response(mock_gitlab_client, glql_analytics_response(sample_data))

    result = await ask_agent(analytics_agent, initial_state, question)

    for tool_name in expected_tools:
        result.assert_has_tool_calls().assert_called_tool(tool_name)
    await result.assert_llm_validates(criteria)
