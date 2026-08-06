# pylint: disable=redefined-outer-name

import json

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.tools.get_glql_schema import _SCHEMAS, GetGlqlSchema


@pytest.fixture
def schema_tool():
    """Fixture providing a GetGlqlSchema tool instance."""
    return GetGlqlSchema(metadata={})


@pytest.mark.asyncio
async def test_single_data_source(schema_tool):
    """Returns schema for a single data source."""
    result = json.loads(await schema_tool._execute(data_source="Pipeline"))
    assert "filters" in result
    assert "display_fields" in result
    assert result["sort_fields"] == []


@pytest.mark.asyncio
async def test_all_data_sources(schema_tool):
    """Returns all schemas when data_source is 'all'."""
    result = json.loads(await schema_tool._execute(data_source="all"))
    assert set(result.keys()) == {
        "WorkItem",
        "MergeRequest",
        "Pipeline",
        "Job",
        "Project",
        "CodeSuggestion",
        "Contribution",
    }


@pytest.mark.asyncio
async def test_comma_separated_sources(schema_tool):
    """Returns multiple schemas for comma-separated input."""
    result = json.loads(await schema_tool._execute(data_source="Pipeline,Job"))
    assert set(result.keys()) == {"Pipeline", "Job"}
    assert "filters" in result["Pipeline"]
    assert "filters" in result["Job"]


@pytest.mark.asyncio
async def test_unknown_data_source(schema_tool):
    """Raises ToolException for unknown data source."""
    with pytest.raises(ToolException) as exc_info:
        await schema_tool._execute(data_source="Unknown")
    assert "Unknown" in str(exc_info.value)


@pytest.mark.asyncio
async def test_unknown_in_comma_separated(schema_tool):
    """Raises ToolException if any comma-separated source is unknown."""
    with pytest.raises(ToolException) as exc_info:
        await schema_tool._execute(data_source="Pipeline,Foo")
    assert "Foo" in str(exc_info.value)


@pytest.mark.asyncio
async def test_default_is_all(schema_tool):
    """Default data_source returns all schemas."""
    result = json.loads(await schema_tool._execute())
    assert len(result) == 7


def test_analytics_field_structure():
    """Analytics fields are either plain strings or well-formed parameterised dicts."""
    for schema_key, schema in _SCHEMAS.items():
        analytics = schema.get("analytics")
        if not analytics:
            continue
        for sub_key in ("dimensions", "metrics"):
            for field in analytics.get(sub_key, []):
                if isinstance(field, str):
                    continue
                assert isinstance(field, dict), (
                    f"Unexpected field type in {schema_key} analytics {sub_key}"
                )
                assert set(field.keys()) == {"name", "parameters"}
                for parameter in field["parameters"]:
                    assert "name" in parameter
                    assert "values" in parameter or "range" in parameter
                    assert "default" in parameter
