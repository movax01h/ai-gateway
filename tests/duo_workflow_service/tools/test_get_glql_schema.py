# pylint: disable=redefined-outer-name

import json

import pytest

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
    """Returns error for unknown data source."""
    result = json.loads(await schema_tool._execute(data_source="Unknown"))
    assert "error" in result
    assert "Unknown" in result["error"]


@pytest.mark.asyncio
async def test_unknown_in_comma_separated(schema_tool):
    """Returns error if any comma-separated source is unknown."""
    result = json.loads(await schema_tool._execute(data_source="Pipeline,Foo"))
    assert "error" in result
    assert "Foo" in result["error"]


@pytest.mark.asyncio
async def test_default_is_all(schema_tool):
    """Default data_source returns all schemas."""
    result = json.loads(await schema_tool._execute())
    assert len(result) == 5


@pytest.mark.asyncio
async def test_schema_structure():
    """Each schema has required keys."""
    for name, schema in _SCHEMAS.items():
        assert "type_values" in schema, f"{name} missing type_values"
        assert "filters" in schema, f"{name} missing filters"
        assert "display_fields" in schema, f"{name} missing display_fields"
        assert "sort_fields" in schema, f"{name} missing sort_fields"
