from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogItemsError,
    CatalogItemsFetchError,
)
from duo_workflow_service.agent_platform.v1.catalog.fetch import fetch_catalog_agents
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse

WORKFLOW_ID = "44"


def _agent_node(**overrides) -> dict:
    return {
        "item_type": "agent",
        "item_id": "53",
        "version": "1.0.0",
        "name": "Code Reviewer",
        "description": "Reviews code.",
        "definition_kind": "inline",
        "definition": {
            "prompt": "You review code.",
            "toolset": ["read_file"],
            "mcp_tools": [],
        },
        **overrides,
    }


def _body(sub_items: list[dict], skipped_reason: str | None = None) -> dict:
    return {
        "schema_version": 1,
        "item": {
            "item_type": "flow",
            "item_id": "7",
            "version": "1.0.0",
            "name": "Parent",
            "description": "A flow.",
            "definition_kind": "inline",
            "definition": {},
            "sub_items": sub_items,
            "sub_items_skipped_reason": skipped_reason,
        },
    }


def _client(status_code: int, body) -> AsyncMock:
    client = AsyncMock()
    client.aget = AsyncMock(
        return_value=GitLabHttpResponse(status_code=status_code, body=body)
    )
    return client


@pytest.mark.asyncio
async def test_requests_the_catalog_items_of_the_run():
    client = _client(200, _body([]))

    await fetch_catalog_agents(client, WORKFLOW_ID)

    client.aget.assert_called_once_with(
        path="/api/v4/ai/duo_workflows/workflows/44/catalog_items",
        parse_json=True,
    )


@pytest.mark.asyncio
async def test_builds_one_agent_per_agent_sub_item():
    client = _client(200, _body([_agent_node(), _agent_node(item_id="54")]))

    fetched = await fetch_catalog_agents(client, WORKFLOW_ID)

    agents = fetched.agents
    assert fetched.skipped_reason is None
    assert [agent.item_id for agent in agents] == ["53", "54"]
    assert agents[0].version == "1.0.0"
    assert agents[0].catalog_name == "Code Reviewer"
    assert agents[0].description == "Reviews code."
    assert agents[0].prompt == "You review code."
    assert agents[0].toolset == ["read_file"]


@pytest.mark.asyncio
async def test_a_missing_toolset_is_empty():
    node = _agent_node(definition={"prompt": "You review code."})
    client = _client(200, _body([node]))

    fetched = await fetch_catalog_agents(client, WORKFLOW_ID)

    assert fetched.agents[0].toolset == []


@pytest.mark.asyncio
async def test_no_sub_items_returns_no_agents():
    client = _client(200, _body([]))

    fetched = await fetch_catalog_agents(client, WORKFLOW_ID)

    assert fetched.agents == []
    assert fetched.skipped_reason is None


@pytest.mark.asyncio
async def test_a_skip_reason_is_returned_with_no_agents():
    client = _client(200, _body([], skipped_reason="custom_agents_disabled"))

    fetched = await fetch_catalog_agents(client, WORKFLOW_ID)

    assert fetched.agents == []
    assert fetched.skipped_reason == "custom_agents_disabled"


@pytest.mark.asyncio
async def test_flow_sub_items_are_ignored():
    flow_node = {"item_type": "flow", "item_id": "9", "version": "1.0.0"}
    client = _client(200, _body([flow_node, _agent_node()]))

    fetched = await fetch_catalog_agents(client, WORKFLOW_ID)

    assert [agent.item_id for agent in fetched.agents] == ["53"]


@pytest.mark.asyncio
async def test_an_error_node_fails_with_the_gitlab_message():
    node = _agent_node(
        definition=None,
        definition_kind=None,
        error={
            "code": "not_visible",
            "message": "Agent 'Code Reviewer' is not visible to flows in this project",
        },
    )
    client = _client(200, _body([node]))

    with pytest.raises(CatalogItemsError, match="is not visible to flows"):
        await fetch_catalog_agents(client, WORKFLOW_ID)


@pytest.mark.asyncio
async def test_an_invalid_agent_fails_with_the_field():
    node = _agent_node(definition={"prompt": "", "toolset": []})
    client = _client(200, _body([node]))

    with pytest.raises(CatalogItemsError, match="prompt"):
        await fetch_catalog_agents(client, WORKFLOW_ID)


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [403, 404, 422])
async def test_an_error_status_fails_with_the_gitlab_message(status_code):
    client = _client(status_code, {"message": "Workflow does not run a saved item"})

    with pytest.raises(
        CatalogItemsFetchError,
        match=f"HTTP {status_code}: Workflow does not run a saved item",
    ):
        await fetch_catalog_agents(client, WORKFLOW_ID)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code,retryable",
    [
        (403, False),
        (404, False),
        (422, False),
        (408, True),
        (429, True),
        (500, True),
        (503, True),
    ],
)
async def test_an_error_status_says_whether_a_retry_can_pass(status_code, retryable):
    """A 404 or 422 gets the same answer on every turn, so it is not retryable."""
    client = _client(status_code, {"message": "nope"})

    with pytest.raises(CatalogItemsFetchError) as exc_info:
        await fetch_catalog_agents(client, WORKFLOW_ID)

    assert exc_info.value.status_code == status_code
    assert exc_info.value.retryable is retryable


@pytest.mark.asyncio
async def test_a_body_that_is_not_an_object_fails():
    client = _client(200, ["not", "an", "object"])

    with pytest.raises(
        CatalogItemsFetchError, match="expected object, got list"
    ) as exc_info:
        await fetch_catalog_agents(client, WORKFLOW_ID)

    assert exc_info.value.status_code is None
    assert exc_info.value.retryable is False


class TestMalformedResponses:
    """What GitLab should never send.

    Each one fails the run with a typed error, never a raw exception.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"item": None},
            {"item": "not an object"},
            {"item": {"sub_items": None}},
            {"item": {"sub_items": "not a list"}},
            {"item": {"sub_items": {"item_type": "agent"}}},
        ],
        ids=[
            "empty",
            "null-item",
            "string-item",
            "null-sub-items",
            "string-sub-items",
            "dict-sub-items",
        ],
    )
    async def test_a_body_without_a_sub_items_list_yields_no_agents(self, body):
        fetched = await fetch_catalog_agents(_client(200, body), WORKFLOW_ID)

        assert fetched.agents == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "node",
        ["a string", 42, None, ["a", "list"]],
        ids=["string", "int", "null", "list"],
    )
    async def test_a_sub_item_that_is_not_an_object_is_skipped(self, node):
        fetched = await fetch_catalog_agents(
            _client(200, _body([node, _agent_node()])), WORKFLOW_ID
        )

        assert [a.item_id for a in fetched.agents] == ["53"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("overrides", "field"),
        [
            ({"item_id": None}, "item_id"),
            ({"item_id": ""}, "item_id"),
            ({"version": None}, "version"),
            ({"name": None}, "name"),
            ({"description": ""}, "description"),
            ({"definition": None}, "prompt"),
            ({"definition": {}}, "prompt"),
            ({"definition": "You review code."}, "prompt"),
            ({"definition": ["You review code."]}, "prompt"),
            ({"definition": {"prompt": "x", "toolset": "read_file"}}, "toolset"),
            ({"definition": {"prompt": "x", "toolset": [1]}}, "toolset"),
        ],
        ids=[
            "null-id",
            "blank-id",
            "null-version",
            "null-name",
            "blank-description",
            "null-definition",
            "empty-definition",
            "string-definition",
            "list-definition",
            "string-toolset",
            "int-tool",
        ],
    )
    async def test_an_agent_missing_a_field_names_it(self, overrides, field):
        client = _client(200, _body([_agent_node(**overrides)]))

        with pytest.raises(CatalogItemsError, match=field):
            await fetch_catalog_agents(client, WORKFLOW_ID)

    @pytest.mark.asyncio
    async def test_an_error_node_without_a_message_still_names_the_agent(self):
        node = _agent_node(definition=None, error={"code": "not_found"})

        with pytest.raises(CatalogItemsError, match="Code Reviewer.*53.*not_found"):
            await fetch_catalog_agents(_client(200, _body([node])), WORKFLOW_ID)

    @pytest.mark.asyncio
    async def test_an_error_node_without_a_name_is_named_by_its_id(self):
        """GitLab leaves the name out when the user may not see the agent."""
        node = {
            "item_type": "agent",
            "item_id": "53",
            "version": "1.0.0",
            "error": {"code": "unknown_item", "message": "Agent 53 was not found"},
        }

        with pytest.raises(CatalogItemsError, match="'53' \\(53\\).*not found"):
            await fetch_catalog_agents(_client(200, _body([node])), WORKFLOW_ID)

    @pytest.mark.asyncio
    async def test_an_error_node_that_is_not_an_object_still_fails(self):
        node = _agent_node(definition=None, error="refused")

        with pytest.raises(CatalogItemsError):
            await fetch_catalog_agents(_client(200, _body([node])), WORKFLOW_ID)

    @pytest.mark.asyncio
    async def test_an_error_status_with_a_non_object_body(self):
        client = _client(500, "<html>Internal Server Error</html>")

        with pytest.raises(CatalogItemsFetchError, match="HTTP 500"):
            await fetch_catalog_agents(client, WORKFLOW_ID)

    @pytest.mark.asyncio
    async def test_an_error_status_with_an_empty_body(self):
        client = _client(502, None)

        with pytest.raises(CatalogItemsFetchError, match="HTTP 502"):
            await fetch_catalog_agents(client, WORKFLOW_ID)

    @pytest.mark.asyncio
    async def test_a_transport_failure_is_not_swallowed(self):
        client = AsyncMock()
        client.aget = AsyncMock(side_effect=ConnectionError("boom"))

        with pytest.raises(ConnectionError):
            await fetch_catalog_agents(client, WORKFLOW_ID)
