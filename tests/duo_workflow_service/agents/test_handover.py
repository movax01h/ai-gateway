from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall

from duo_workflow_service.agents import HandoverAgent
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    Plan,
    ToolStatus,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.tools.handover import HandoverTool


class TestHandoverAgent:
    @pytest.mark.asyncio
    @patch("duo_workflow_service.agents.handover.datetime")
    async def test_run_set_status(self, mock_datetime, workflow_state):
        mock_datetime.now.return_value = datetime(
            2025, 1, 1, 12, 0, tzinfo=timezone.utc
        )
        mock_datetime.timezone = timezone

        assert await HandoverAgent(
            new_status=WorkflowStatusEnum.COMPLETED, handover_from="test_agent"
        ).run(workflow_state) == {
            "status": WorkflowStatusEnum.COMPLETED,
            "ui_chat_log": [],
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "handover_from, conversation_history, include_conversation_history, expected_handover, expected_ui_chat_log",
        [
            (
                "test_agent_executor",
                [
                    AIMessage(
                        id="1",
                        content="test_message",
                        tool_calls=[
                            ToolCall(
                                id="1",
                                name=str(HandoverTool.tool_title),
                                args={"summary": "This is summary"},
                            )
                        ],
                    )
                ],
                True,
                [],
                [],
            ),
            (
                "test_agent",
                [
                    HumanMessage(
                        id="1",
                        content="test message",
                    )
                ],
                False,
                [],
                [],
            ),
            (
                "test_agent",
                [
                    AIMessage(
                        id="1",
                        content="test_message",
                        tool_calls=[
                            ToolCall(
                                id="1",
                                name="read_file_tool",
                                args={},
                            )
                        ],
                    ),
                    AIMessage(
                        id="ai-msg-id",
                        content="test_message",
                        tool_calls=[
                            ToolCall(
                                id="2",
                                name=str(HandoverTool.tool_title),
                                args={"summary": "This is awesome summary"},
                            )
                        ],
                    ),
                ],
                True,
                [
                    AIMessage(
                        content="This is awesome summary",
                    ),
                ],
                [
                    {
                        "content": "This is awesome summary",
                        "correlation_id": None,
                        "message_type": MessageTypeEnum.AGENT,
                        "message_sub_type": None,
                        "status": ToolStatus.SUCCESS,
                        "timestamp": "2025-01-01T12:00:00+00:00",
                        "tool_info": None,
                        "additional_context": None,
                        "message_id": "ai-msg-id",
                    },
                ],
            ),
            (
                "test_agent",
                [
                    AIMessage(
                        id="ai-msg-id",
                        content="test_message",
                        tool_calls=[
                            ToolCall(
                                id="1",
                                name=str(HandoverTool.tool_title),
                                args={"summary": "This is summary"},
                            )
                        ],
                    )
                ],
                True,
                [AIMessage(content="This is summary")],
                [
                    {
                        "message_type": MessageTypeEnum.AGENT,
                        "message_sub_type": None,
                        "content": "This is summary",
                        "timestamp": "2025-01-01T12:00:00+00:00",
                        "status": ToolStatus.SUCCESS,
                        "correlation_id": None,
                        "tool_info": None,
                        "additional_context": None,
                        "message_id": "ai-msg-id",
                    },
                ],
            ),
            (
                "test_agent",
                [
                    SystemMessage(
                        content="You are AGI prepare the answer to life the universe and everything",
                    ),
                    AIMessage(id="1", content="42"),
                ],
                True,
                [],
                [],
            ),
            (
                "test_agent",
                [
                    AIMessage(
                        id="1",
                        content="This is the analysis result",
                        tool_calls=[
                            ToolCall(
                                id="1",
                                name=str(HandoverTool.tool_title),
                                args={},  # Empty arguments
                            )
                        ],
                    )
                ],
                True,
                [],
                [],
            ),
            (
                "test_agent",
                [
                    AIMessage(
                        id="1",
                        content="This is the analysis result",
                        tool_calls=[
                            ToolCall(
                                id="1",
                                name=str(HandoverTool.tool_title),
                                args={"summary": ""},  # Empty summary value
                            )
                        ],
                    )
                ],
                True,
                [],
                [],
            ),
        ],
    )
    @patch("duo_workflow_service.agents.handover.datetime")
    async def test_run_handover(
        self,
        mock_datetime,
        handover_from,
        conversation_history,
        include_conversation_history,
        expected_handover,
        expected_ui_chat_log,
    ):
        mock_datetime.now.return_value = datetime(
            2025, 1, 1, 12, 0, tzinfo=timezone.utc
        )
        mock_datetime.timezone = timezone

        state = WorkflowState(
            plan=Plan(steps=[]),
            status=WorkflowStatusEnum.NOT_STARTED,
            handover=[],
            conversation_history={"test_agent": conversation_history},
            last_human_input=None,
            ui_chat_log=[
                {
                    "message_type": MessageTypeEnum.AGENT,
                    "message_sub_type": None,
                    "content": "This is summary",
                    "timestamp": "2025-01-08T12:00:00Z",
                    "status": ToolStatus.SUCCESS,
                    "correlation_id": None,
                    "tool_info": None,
                    "additional_context": None,
                    "message_id": "start-id",
                }
            ],
        )
        overwrites = await HandoverAgent(
            new_status=WorkflowStatusEnum.COMPLETED,
            handover_from=handover_from,
            include_conversation_history=include_conversation_history,
        ).run(state)

        expected_result = {
            "status": WorkflowStatusEnum.COMPLETED,
            "ui_chat_log": expected_ui_chat_log,
        }
        if expected_handover:
            expected_result["handover"] = expected_handover

        assert overwrites == expected_result

    # pylint: enable=too-many-positional-arguments
