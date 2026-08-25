import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.fetch_glql_schema import (
    MAX_SCHEMA_BYTES,
    SCHEMA_PATH,
    FetchGlqlSchema,
    FetchGlqlSchemaInput,
)
from lib.context import gitlab_version

# A cut-down copy of what the endpoint serves, keeping one of every shape the
# agent prompt tells the model to read.
DOCUMENT = {
    "sources": [
        {
            "name": "WorkItems",
            "label": "work items",
            "modes": [
                {
                    "mode": "Standard",
                    "allowed_scopes": ["project"],
                    "filter_fields": [
                        {
                            "name": "type",
                            "value_types": [
                                {
                                    "kind": "Enum",
                                    "operators": ["="],
                                    "values": ["Issue", "Epic"],
                                }
                            ],
                        },
                        {
                            "name": "label",
                            "aliases": ["labels"],
                            "value_types": [{"kind": "String", "operators": ["="]}],
                        },
                    ],
                    "display_fields": [{"name": "title"}],
                    "sort_fields": ["created"],
                }
            ],
        }
    ],
    "operators": [{"symbol": "=", "name": "Equal", "label": "equals"}],
    "value_kinds": [{"name": "String", "description": "A quoted string."}],
    "reference_types": [{"name": "LabelRef", "symbol": "~", "example": "~frontend"}],
    "display_types": [{"name": "list", "description": "A bulleted list of items."}],
    "functions": [
        {
            "name": "today",
            "kind": "value",
            "description": "Today.",
            "args": [],
            "returns": "Date",
        }
    ],
    "version": "0.34.0",
}


def ok(body):
    return GitLabHttpResponse(status_code=200, body=json.dumps(body))


@pytest.fixture(name="gitlab_client")
def gitlab_client_fixture():
    client = AsyncMock()
    client.aget = AsyncMock(return_value=ok(DOCUMENT))
    return client


@pytest.fixture(name="report_version", autouse=True)
def report_version_fixture():
    """Set the version the tool sees, defaulting to one that has the endpoint.

    `version_compatibility` decides and this tool names the version in the
    error, but both read the same context var, so one set covers both.
    """
    token = gitlab_version.set("19.3.0")

    yield gitlab_version.set

    gitlab_version.reset(token)


@pytest.fixture(name="schema_tool")
def schema_tool_fixture(gitlab_client):
    return FetchGlqlSchema(metadata={"gitlab_client": gitlab_client})


@pytest.mark.asyncio
async def test_returns_the_whole_document(schema_tool, gitlab_client):
    result = await schema_tool._execute()

    gitlab_client.aget.assert_awaited_once()
    call = gitlab_client.aget.await_args
    assert call.kwargs["path"] == SCHEMA_PATH
    # No parameters: the endpoint has none.
    assert "params" not in call.kwargs
    # The instance's body verbatim, not a re-serialisation of it.
    assert result == json.dumps(DOCUMENT)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        # The client can hand back raw bytes...
        json.dumps(DOCUMENT).encode("utf-8"),
        # ...or an already-parsed document despite parse_json=False.
        DOCUMENT,
    ],
    ids=["bytes", "dict"],
)
async def test_normalises_non_string_bodies(schema_tool, gitlab_client, body):
    gitlab_client.aget.return_value = GitLabHttpResponse(status_code=200, body=body)

    result = await schema_tool._execute()

    assert result == json.dumps(DOCUMENT)


@pytest.mark.asyncio
async def test_caches_for_the_run(schema_tool, gitlab_client):
    await schema_tool._execute()
    await schema_tool._execute()
    await schema_tool._execute()

    gitlab_client.aget.assert_awaited_once()


@pytest.mark.asyncio
async def test_parallel_calls_fetch_once(schema_tool, gitlab_client):
    """A model batching parallel calls should still cost one request.

    The stub has to yield, or every call runs to completion before the next starts and the test passes whether or not
    the fetch is guarded.
    """

    async def slow_aget(**_kwargs):
        await asyncio.sleep(0)
        return ok(DOCUMENT)

    gitlab_client.aget = AsyncMock(side_effect=slow_aget)

    results = await asyncio.gather(*(schema_tool._execute() for _ in range(3)))

    gitlab_client.aget.assert_awaited_once()
    assert [json.loads(result) for result in results] == [DOCUMENT] * 3


@pytest.mark.asyncio
async def test_ignores_unexpected_arguments(schema_tool):
    """`get_glql_schema` took a `data_source`; a model may still send one."""
    result = json.loads(await schema_tool._execute(data_source="Pipelines"))

    assert result == DOCUMENT


class TestErrors:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response,match",
        [
            # Nothing to fall back to: only instances with the endpoint get this
            # tool, so a 404 is a broken pairing rather than an old instance.
            (GitLabHttpResponse(status_code=404, body="404 Not Found"), "404"),
            (GitLabHttpResponse(status_code=500, body="boom"), "HTTP 500"),
            (
                GitLabHttpResponse(status_code=200, body="not json"),
                "not valid JSON",
            ),
            (ok({"operators": []}), "did not contain any sources"),
            (ok([]), "did not contain any sources"),
        ],
    )
    async def test_a_bad_response_raises(
        self, schema_tool, gitlab_client, response, match
    ):
        gitlab_client.aget.return_value = response

        with pytest.raises(ToolException, match=match):
            await schema_tool._execute()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reported",
        [
            "18.6.0",
            "19.2.0",
            "19.2.0-pre",
            "19.2.9",
            # Not valid PEP 440 and no leading X.Y(.Z) to salvage, so parsing
            # falls back to the 18.6.0 default.
            "garbage",
            None,
        ],
    )
    async def test_an_instance_without_the_endpoint_raises(
        self, schema_tool, gitlab_client, report_version, reported
    ):
        """Say the version is too old rather than let the endpoint 404."""
        report_version(reported)

        with pytest.raises(ToolException, match="GitLab 19.3.0 and later"):
            await schema_tool._execute()

        gitlab_client.aget.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reported",
        [
            "19.3.0-pre",
            "19.3.0",
            "19.4.0",
            # Two components: the header GitLab.com sometimes sends.
            "19.3",
            # Not valid PEP 440, but the leading 19.3.0 is; a GDK reports this.
            "19.3.0-pre-g9262b86a1",
        ],
    )
    async def test_an_instance_with_the_endpoint_is_allowed(
        self, schema_tool, report_version, reported
    ):
        report_version(reported)

        assert json.loads(await schema_tool._execute()) == DOCUMENT

    @pytest.mark.asyncio
    async def test_an_oversized_response_raises_and_is_not_cached(
        self, schema_tool, gitlab_client
    ):
        gitlab_client.aget.return_value = GitLabHttpResponse(
            status_code=200, body="x" * (MAX_SCHEMA_BYTES + 1)
        )

        with pytest.raises(ToolException, match="too large"):
            await schema_tool._execute()

        gitlab_client.aget.return_value = ok(DOCUMENT)

        assert json.loads(await schema_tool._execute()) == DOCUMENT

    @pytest.mark.asyncio
    async def test_a_failure_is_not_cached(self, schema_tool, gitlab_client):
        """A retry should reach the instance again rather than repeat the error."""
        gitlab_client.aget.return_value = GitLabHttpResponse(
            status_code=500, body="boom"
        )

        with pytest.raises(ToolException):
            await schema_tool._execute()

        gitlab_client.aget.return_value = ok(DOCUMENT)

        assert json.loads(await schema_tool._execute()) == DOCUMENT


@pytest.mark.asyncio
async def test_format_display_message(schema_tool):
    assert (
        schema_tool.format_display_message(FetchGlqlSchemaInput())
        == "Looking up the GLQL schema"
    )
