import base64
import json
import zlib
from unittest.mock import AsyncMock, Mock

import pytest
from google.protobuf import struct_pb2
from langgraph.checkpoint.base import CheckpointMetadata

from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    FlowConfig as V1FlowConfig,
)
from duo_workflow_service.checkpointer.content_retention import (
    is_metadata_only,
    reduce_ui_chat_log,
    reduce_ui_chat_log_entry,
)
from duo_workflow_service.checkpointer.gitlab_workflow import GitLabWorkflow
from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    WorkflowStatusEventEnum,
    uncompress_checkpoint,
)
from duo_workflow_service.checkpointer.notifier import UserInterface
from duo_workflow_service.entities.state import WorkflowStatusEnum
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.workflows.registry import _load_flow_from_inline_config

SECRET = "SECRET-MODEL-TEXT"

KEPT_ENTRY_FIELDS = {
    "message_id": "msg-1",
    "message_type": "tool",
    "message_sub_type": "read_file",
    "status": "success",
    "timestamp": "2026-10-06T00:00:00+00:00",
    "correlation_id": "corr-1",
    "component_name": "triage",
}


def _chat_entry(message_id="msg-1"):
    return {
        **KEPT_ENTRY_FIELDS,
        "message_id": message_id,
        "content": SECRET,
        "tool_info": {
            "name": "read_file",
            "args": {"file_path": SECRET},
            "tool_response": SECRET,
        },
        "additional_context": [{"content": SECRET}],
        "subsession_id": SECRET,
    }


def _reduced_entry(message_id="msg-1"):
    return {
        **KEPT_ENTRY_FIELDS,
        "message_id": message_id,
        "content": "",
        "additional_context": None,
        "tool_info": {"name": "read_file", "args": {}},
    }


def _assert_readers_accept(entry):
    """Mirror the checks the chat-log readers apply to each entry.

    duo-cli: gitlab-lsp packages/core/workflow_api/src/ui_chat_log.ts ``ChatLogSchema`` (zod 4, so a
    ``z.unknown()`` key must be present). Session page: duo-ui ``DuoToolMessage`` ``message`` prop validator
    ``hasToolInfo``.
    """
    nullable_str = (str, type(None))
    assert isinstance(entry["content"], str)
    assert isinstance(entry["timestamp"], str)
    assert isinstance(entry["message_sub_type"], nullable_str)
    assert isinstance(entry["status"], nullable_str)
    assert isinstance(entry["correlation_id"], nullable_str)
    assert "additional_context" in entry
    if "message_id" in entry:
        assert isinstance(entry["message_id"], str)
    assert isinstance(entry.get("component_name"), nullable_str)

    tool_info = entry["tool_info"]
    if tool_info is not None:
        assert isinstance(tool_info["name"], str)
        assert isinstance(tool_info["args"], dict)
    if entry["message_type"] in ("user", "agent"):
        assert tool_info is None
    elif entry["message_type"] == "request":
        assert tool_info is not None or entry["message_sub_type"] == "approval"
    elif entry["message_type"] == "tool":
        assert isinstance(tool_info, dict)  # DuoToolMessage hasToolInfo
    else:
        pytest.fail(f"unexpected message_type {entry['message_type']}")


def _checkpoint(checkpoint_id, messages=1):
    channel_values = {
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {"agent": [f"{SECRET} {i}" for i in range(messages)]},
        "context": {"triage": {"reasoning": SECRET}},
        "agent_context_limits": {"agent": 1000},
        "ui_chat_log": [_chat_entry(f"msg-{i}") for i in range(messages)],
        "__pregel_tasks": [SECRET],
    }
    return {
        "v": 1,
        "id": checkpoint_id,
        "ts": "2026-10-06T00:00:00+00:00",
        "channel_values": channel_values,
        "channel_versions": dict.fromkeys(channel_values, messages),
        "versions_seen": {},
        "updated_channels": None,
    }


def _metadata():
    metadata = CheckpointMetadata(source="loop", step=3)
    metadata["writes"] = {"triage": {"reasoning": SECRET}}
    return metadata


def _decoded_posts(http_client) -> list[dict]:
    """Every checkpoint POST body, with the compressed snapshot and blobs decoded."""
    posts = []
    for call in http_client.apost.call_args_list:
        body = json.loads(call.kwargs["body"])
        if "compressed_checkpoint" in body:
            body["compressed_checkpoint"] = uncompress_checkpoint(
                body["compressed_checkpoint"]
            )
        for blob in body.get("channel_blobs", []):
            blob["data"] = json.loads(zlib.decompress(base64.b64decode(blob["data"])))
        posts.append(body)
    return posts


@pytest.fixture(autouse=True)
def prepare_container(  # pylint: disable=unused-argument
    mock_duo_workflow_service_container,
):
    pass


@pytest.fixture(name="http_client")
def http_client_fixture():
    client = AsyncMock()
    client.apost.return_value = GitLabHttpResponse(status_code=200, body={})
    client.apatch.return_value = GitLabHttpResponse(status_code=200, body={})
    return client


@pytest.fixture(name="workflow_config")
def workflow_config_fixture():
    return {
        "first_checkpoint": None,
        "latest_checkpoint": None,
        "workflow_status": "created",
        "incremental_checkpoints_enabled": False,
        "archived": False,
    }


@pytest.fixture(name="checkpointer")
def checkpointer_fixture(http_client, workflow_id, workflow_type, workflow_config):
    # Built exactly as a metadata-retention flow builds it: no retention argument.
    return GitLabWorkflow(http_client, workflow_id, workflow_type, workflow_config)


@pytest.mark.parametrize(
    ("content_retention", "expected"),
    [("metadata", True), ("full", False), (None, False)],
)
def test_is_metadata_only(content_retention, expected):
    assert is_metadata_only(content_retention) is expected


class TestReduction:
    def test_reduces_every_chat_log_entry(self):
        assert reduce_ui_chat_log([_chat_entry("msg-0"), _chat_entry("msg-1")]) == [
            _reduced_entry("msg-0"),
            _reduced_entry("msg-1"),
        ]
        assert not reduce_ui_chat_log(None)

    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"message_type": "tool", "tool_info": None, "message_sub_type": None},
            {"message_type": "agent", "tool_info": None, "message_sub_type": None},
            {"message_type": "user", "tool_info": None, "message_sub_type": None},
            {"message_type": "request", "message_sub_type": None},
            {
                "message_type": "request",
                "tool_info": None,
                "message_sub_type": "approval",
            },
            {"message_id": None, "correlation_id": None, "status": None},
        ],
    )
    def test_reduced_entry_passes_the_chat_log_readers(self, overrides):
        reduced = reduce_ui_chat_log_entry({**_chat_entry(), **overrides})

        _assert_readers_accept(reduced)
        assert SECRET not in json.dumps(reduced)

    def test_message_and_approval_entries_keep_no_tool_info(self):
        for message_type, sub_type in (("agent", None), ("request", "approval")):
            entry = {
                **_chat_entry(),
                "message_type": message_type,
                "message_sub_type": sub_type,
                "tool_info": None,
            }

            assert reduce_ui_chat_log_entry(entry)["tool_info"] is None

    def test_tool_entry_without_tool_info_gets_an_empty_one(self):
        entry = {**_chat_entry(), "tool_info": None}

        assert reduce_ui_chat_log_entry(entry)["tool_info"] == {"name": "", "args": {}}

    def test_missing_message_id_is_omitted_not_null(self):
        entry = {**_chat_entry(), "message_id": None}

        assert "message_id" not in reduce_ui_chat_log_entry(entry)


class TestFlowConfig:
    def test_bl_security_declares_metadata_retention(self):
        assert (
            FlowConfig.from_yaml_config("bl_security", "1.0.0").content_retention
            == "metadata"
        )

    def test_every_bl_security_version_declares_metadata_retention(self):
        versions = sorted((FlowConfig.DIRECTORY_PATH / "bl_security").glob("*.yml"))

        assert versions
        for path in versions:
            config = FlowConfig.from_yaml_config("bl_security", path.stem)
            assert config.content_retention == "metadata", path.name

    def test_defaults_to_full(self):
        config = FlowConfig(
            flow={}, components=[], routers=[], environment="ambient", version="v1"
        )

        assert config.content_retention == "full"

    @pytest.mark.parametrize(
        "environment,component",
        [
            ("chat", {"name": "a", "type": "AgentComponent"}),
            ("ambient", {"name": "h", "type": "HumanInputComponent"}),
            (
                "ambient",
                {"name": "a", "type": "AgentComponent", "require_tool_approval": True},
            ),
        ],
    )
    def test_metadata_is_allowed_for_a_flow_that_can_pause(
        self, environment, component
    ):
        # Checkpoints stay complete, so a paused session can resume.
        config = FlowConfig(
            flow={},
            components=[component],
            routers=[],
            environment=environment,
            version="experimental",
            content_retention="metadata",
        )

        assert config.content_retention == "metadata"

    def test_v1_flow_configs_ignore_it(self):
        config = V1FlowConfig(
            flow={},
            components=[],
            routers=[],
            environment="ambient",
            version="v1",
            content_retention="metadata",
        )

        assert not hasattr(config, "content_retention")

    @pytest.mark.parametrize("schema_version", ["experimental", "v1"])
    def test_an_inline_config_may_not_set_it(self, schema_version):
        struct = struct_pb2.Struct()
        struct.update(
            {
                "version": schema_version,
                "environment": "ambient",
                "components": [],
                "routers": [],
                "flow": {"entry_point": "a"},
                "content_retention": "metadata",
            }
        )

        with pytest.raises(ValueError, match="only supported for bundled flows"):
            _load_flow_from_inline_config(struct, schema_version)


class TestCheckpointsStayComplete:
    """GitLab reduces checkpoints on read; DWS must save them whole so the session can resume."""

    @pytest.mark.asyncio
    async def test_snapshot_and_metadata_keep_the_conversation(
        self, checkpointer, http_client
    ):
        await checkpointer.aput(
            {"configurable": {"checkpoint_id": "0"}}, _checkpoint("1"), _metadata(), {}
        )

        (post,) = _decoded_posts(http_client)
        channel_values = post["compressed_checkpoint"]["channel_values"]
        assert channel_values["conversation_history"] == {"agent": [f"{SECRET} 0"]}
        assert channel_values["ui_chat_log"][0]["content"] == SECRET
        assert post["metadata"]["writes"] == {"triage": {"reasoning": SECRET}}

    @pytest.mark.asyncio
    async def test_interrupt_writes_are_persisted(self, checkpointer, http_client):
        await checkpointer.aput_writes(
            {"configurable": {"checkpoint_id": "1", "thread_id": "123"}},
            [("__interrupt__", SECRET)],
            "task",
        )

        http_client.apost.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "workflow_status,status_event",
        [
            ("failed", WorkflowStatusEventEnum.RETRY),
            ("input_required", WorkflowStatusEventEnum.RESUME),
        ],
    )
    async def test_a_saved_session_resumes(
        self, checkpointer, http_client, workflow_config, workflow_status, status_event
    ):
        workflow_config["workflow_status"] = workflow_status
        workflow_config["first_checkpoint"] = {"checkpoint": "{}"}

        await checkpointer.__aenter__()

        assert checkpointer.initial_status_event == status_event
        assert json.dumps(
            {"status_event": WorkflowStatusEventEnum.DROP.value}
        ) not in str(http_client.apatch.call_args_list)


class TestLiveStream:
    def test_checkpoint_carries_no_text(self):
        notifier = UserInterface(
            outbox=Mock(), goal=SECRET, content_retention="metadata"
        )
        notifier.status = WorkflowStatusEnum.EXECUTION
        notifier.ui_chat_log = [_chat_entry()]
        notifier.steps = [{"id": "1", "description": SECRET, "status": "In Progress"}]

        checkpoint = notifier.most_recent_new_checkpoint()

        assert checkpoint.goal == ""
        assert json.loads(checkpoint.checkpoint)["channel_values"] == {
            "ui_chat_log": [_reduced_entry()],
            "plan": {"steps": [{"id": "1", "status": "In Progress"}]},
        }
