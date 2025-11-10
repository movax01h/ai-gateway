import json
from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.previous_context import (
    GetSessionContext,
    GetSessionContextInput,
)


@pytest.fixture(name="gitlab_client")
def gitlab_client_fixture():
    client = AsyncMock()
    return client


@pytest.fixture(name="get_last_checkpoint_tool")
def get_last_checkpoint_tool_fixture(gitlab_client):
    return GetSessionContext(
        metadata={"gitlab_client": gitlab_client, "gitlab_host": "gitlab.example.com"}
    )


class TestGetSessionContext:
    def test_format_display_message(self, get_last_checkpoint_tool):
        args = GetSessionContextInput(previous_session_id=123)
        result = get_last_checkpoint_tool.format_display_message(args)

        assert result == "Get context for session 123"

    @pytest.mark.asyncio
    async def test_arun_success(self, get_last_checkpoint_tool, gitlab_client):
        mock_checkpoint = {
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
            "metadata": {
                "step": 4,
                "source": "loop",
                "writes": {},
                "parents": {},
                "thread_id": "123",
            },
        }
        mock_response = GitLabHttpResponse(
            status_code=200,
            body=[mock_checkpoint],
        )
        gitlab_client.aget.return_value = mock_response

        result = await get_last_checkpoint_tool._arun(previous_session_id=123)

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
        mock_response = GitLabHttpResponse(
            status_code=200,
            body=[],
        )
        gitlab_client.aget.return_value = mock_response

        result = await get_last_checkpoint_tool._arun(previous_session_id=123)

        # Verify the error message
        parsed_result = json.loads(result)
        assert "error" in parsed_result
        assert parsed_result["error"] == "Unable to find checkpoint for this session"

    @pytest.mark.asyncio
    async def test_arun_api_error(self, get_last_checkpoint_tool, gitlab_client):
        mock_response = GitLabHttpResponse(
            status_code=404,
            body={"message": "unexpected status code: 404"},
        )
        gitlab_client.aget.return_value = mock_response

        result = await get_last_checkpoint_tool._arun(previous_session_id=123)

        # Verify the error message
        parsed_result = json.loads(result)
        assert "error" in parsed_result
        assert parsed_result["error"] == "API Error"

    @pytest.mark.asyncio
    async def test_format_checkpoint_context_no_checkpoint_data(
        self, get_last_checkpoint_tool
    ):
        checkpoint = {
            "checkpoint": {},
            "metadata": {
                "step": 4,
                "source": "loop",
                "writes": {},
                "parents": {},
                "thread_id": "123",
            },
        }

        context_str = get_last_checkpoint_tool._format_checkpoint_context(checkpoint)

        context = json.loads(context_str)
        assert isinstance(context, dict)
        assert context["workflow"]["id"] == "123"
        assert context["workflow"]["plan"] == {"steps": []}
        assert context["workflow"]["goal"] == "No goal available"
        assert context["workflow"]["summary"] == "No summary available"

    @pytest.mark.asyncio
    async def test_format_checkpoint_context_invalid_data(
        self, get_last_checkpoint_tool
    ):
        invalid_checkpoint = {
            "checkpoint": {"invalid": "data"},
            "metadata": {
                "step": 4,
                "source": "loop",
                "writes": {},
                "parents": {},
                "thread_id": "123",
            },
        }

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
            "checkpoint": {
                "channel_values": {"handover": "not-a-list", "status": "Completed"}
            },
            "metadata": {
                "step": 4,
                "source": "loop",
                "writes": {},
                "parents": {},
                "thread_id": "123",
            },
        }

        with pytest.raises(
            ValueError,
            match="Unable to parse context from last checkpoint for this session",
        ):
            get_last_checkpoint_tool._format_checkpoint_context(bad_checkpoint)

    @pytest.mark.asyncio
    async def test_format_checkpoint_context_failed_workflow(
        self, get_last_checkpoint_tool
    ):
        checkpoint = {
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
            "metadata": {
                "step": 4,
                "source": "loop",
                "writes": {},
                "parents": {},
                "thread_id": "123",
            },
        }

        with pytest.raises(
            ValueError,
            match="Can only collect context on completed workflows",
        ):
            get_last_checkpoint_tool._format_checkpoint_context(checkpoint)
