# pylint: disable=file-naming-for-tests
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from ai_gateway.models.v2.openai import ChatOpenAI


class ReadFileArgs(BaseModel):
    path: str
    limit: int = 20


MCP_TOOL = StructuredTool(
    name="list_merge_requests",
    description="List merge requests.",
    args_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "state": {"type": "string", "enum": ["opened", "closed"]},
        },
        "required": ["project_id"],
    },
    func=lambda **_: "",
)
NATIVE_TOOL = StructuredTool(
    name="read_file",
    description="Read a file.",
    args_schema=ReadFileArgs,
    func=lambda **_: "",
)


def _request_tools(model: ChatOpenAI, **kwargs) -> dict:
    bound = model.bind_tools([MCP_TOOL, NATIVE_TOOL], **kwargs)
    assert isinstance(bound, RunnableBinding)
    payload = model._get_request_payload([HumanMessage("hi")], **bound.kwargs)
    return {tool.get("name", tool["type"]): tool for tool in payload["tools"]}


@pytest.fixture(name="model")
def model_fixture():
    return ChatOpenAI(
        model="gpt-5.6-sol", openai_api_key="dummy", output_version="responses/v1"
    )


class TestNonStrictTools:
    def test_function_tools_are_sent_with_strict_false(self, model):
        tools = _request_tools(model)

        assert tools["list_merge_requests"]["strict"] is False
        assert tools["read_file"]["strict"] is False

    def test_schemas_are_sent_unchanged(self, model):
        tools = _request_tools(model)

        assert tools["list_merge_requests"]["parameters"]["properties"]["state"] == {
            "type": "string",
            "enum": ["opened", "closed"],
        }

    def test_built_in_web_search_is_untouched(self, model):
        tools = _request_tools(model, web_search_options={})

        assert "strict" not in tools["web_search"]

    def test_an_explicit_strict_is_kept(self, model):
        tools = _request_tools(model, strict=True)

        assert tools["list_merge_requests"]["strict"] is True
