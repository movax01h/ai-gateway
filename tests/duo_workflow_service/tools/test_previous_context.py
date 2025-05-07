import json
from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.tools.previous_context import (
    GetWorkflowContext,
    GetWorkflowContextInput,
)


@pytest.fixture
def gitlab_client():
    client = AsyncMock()
    return client


@pytest.fixture
def get_last_checkpoint_tool(gitlab_client):
    return GetWorkflowContext(
        metadata={"gitlab_client": gitlab_client, "gitlab_host": "gitlab.example.com"}
    )


class TestGetWorkflowContext:
    def test_format_display_message(self, get_last_checkpoint_tool):
        args = GetWorkflowContextInput(workflow_id="123")
        result = get_last_checkpoint_tool.format_display_message(args)

        assert result == "Get context for workflow 123"

    @pytest.mark.asyncio
    async def test_arun_success(self, get_last_checkpoint_tool, gitlab_client):
        mock_checkpoint = {
            "workflow_id": "123",
            "checkpoint": {
                "channel_values": {
                    "plan": {
                        "steps": [
                            {"id": "1", "description": "Task 1", "status": "Completed"}
                        ]
                    },
                    "status": "Completed",
                    "handover": [
                        {"type": "system", "content": "System message"},
                        {"type": "human", "content": "Your goal is: Create a feature"},
                        {"type": "ai", "content": "Task summary"},
                    ],
                }
            },
        }
        gitlab_client.aget.return_value = [mock_checkpoint]

        result = await get_last_checkpoint_tool._arun(workflow_id="123")

        gitlab_client.aget.assert_called_once_with(
            path="/api/v4/ai/duo_workflows/workflows/123/checkpoints?per_page=1",
            parse_json=True,
        )

        parsed_result = json.loads(result)
        assert "context" in parsed_result
        formatted_checkpoint = parsed_result["context"]

        context_dict = json.loads(formatted_checkpoint)
        assert isinstance(context_dict, dict)
        assert "workflow" in context_dict

        workflow = context_dict["workflow"]
        assert workflow["id"] == "123"
        assert workflow["goal"] == "Create a feature"
        assert workflow["summary"] == "Task summary"
        assert "plan" in workflow

    @pytest.mark.asyncio
    async def test_arun_empty_response(self, get_last_checkpoint_tool, gitlab_client):
        gitlab_client.aget.return_value = []

        result = await get_last_checkpoint_tool._arun(workflow_id="123")

        # Verify the error message
        parsed_result = json.loads(result)
        assert "error" in parsed_result
        assert parsed_result["error"] == "No checkpoints found for this workflow"

    @pytest.mark.asyncio
    async def test_arun_api_error(self, get_last_checkpoint_tool, gitlab_client):
        gitlab_client.aget.side_effect = Exception("API Error")

        result = await get_last_checkpoint_tool._arun(workflow_id="123")

        # Verify the error message
        parsed_result = json.loads(result)
        assert "error" in parsed_result
        assert parsed_result["error"] == "API Error"

    @pytest.mark.asyncio
    async def test_format_checkpoint_context_no_checkpoint_data(
        self, get_last_checkpoint_tool
    ):
        checkpoint = {"workflow_id": "123"}

        context_str = get_last_checkpoint_tool._format_checkpoint_context(checkpoint)

        context = json.loads(context_str)
        assert isinstance(context, dict)
        print(context)
        assert context["workflow"]["id"] == "123"
        assert context["workflow"]["plan"] == {"steps": []}
        assert context["workflow"]["goal"] == "No goal available"
        assert context["workflow"]["summary"] == "No summary available"

    @pytest.mark.asyncio
    async def test_format_checkpoint_context_invalid_data(
        self, get_last_checkpoint_tool
    ):
        invalid_checkpoint = {"workflow_id": "123", "checkpoint": {"invalid": "data"}}

        context_str = get_last_checkpoint_tool._format_checkpoint_context(
            invalid_checkpoint
        )

        context = json.loads(context_str)
        assert isinstance(context, dict)
        assert context["workflow"]["id"] == "123"
        assert context["workflow"]["plan"] == {"steps": []}
        assert context["workflow"]["goal"] == "No goal available"
        assert context["workflow"]["summary"] == "No summary available"

    @pytest.mark.asyncio
    async def test_format_checkpoint_context_parsing_exception(
        self, get_last_checkpoint_tool
    ):
        # Create a checkpoint that will cause an exception during parsing
        bad_checkpoint = {
            "workflow_id": "123",
            "checkpoint": {
                "channel_values": {"handover": "not-a-list", "status": "Completed"}
            },
        }

        with pytest.raises(
            ValueError,
            match="Unable to parse context from last checkpoint for this workflow",
        ):
            get_last_checkpoint_tool._format_checkpoint_context(bad_checkpoint)

    @pytest.mark.asyncio
    async def test_format_checkpoint_context_failed_workflow(
        self, get_last_checkpoint_tool
    ):
        checkpoint = {
            "workflow_id": "123",
            "checkpoint": {
                "channel_values": {
                    "plan": {"steps": []},
                    "handover": [
                        {"type": "system", "content": "System message"},
                        {"type": "human", "content": "Message without goal"},
                        {"type": "ai", "content": "Summary"},
                    ],
                    "status": "Failed",
                }
            },
        }

        with pytest.raises(
            ValueError,
            match="Can only collect context on completed workflows",
        ):
            get_last_checkpoint_tool._format_checkpoint_context(checkpoint)
