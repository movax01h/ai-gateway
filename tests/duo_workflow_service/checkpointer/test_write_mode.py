import pytest
from langgraph.types import Overwrite

from duo_workflow_service.checkpointer.write_mode import (
    compaction_ui_chat_log_update,
    incremental_checkpoints_enabled,
    write_incremental_only,
)
from lib.context import client_capabilities, gitlab_version


def test_write_incremental_only_is_off_by_default():
    assert write_incremental_only() is False


@pytest.mark.usefixtures("incremental_checkpoints_only")
def test_write_incremental_only_needs_flag_and_capability():
    assert write_incremental_only() is True

    enabled_token = incremental_checkpoints_enabled.set(False)
    try:
        assert write_incremental_only() is False
    finally:
        incremental_checkpoints_enabled.reset(enabled_token)

    capabilities_token = client_capabilities.set(set())
    try:
        assert write_incremental_only() is False
    finally:
        client_capabilities.reset(capabilities_token)

    version_token = gitlab_version.set("18.6.0")
    try:
        assert write_incremental_only() is False
    finally:
        gitlab_version.reset(version_token)


@pytest.mark.usefixtures("incremental_checkpoints_only")
def test_write_incremental_only_explicit_flag_wins_over_context():
    """The checkpointer passes its own workflow flag instead of the published one."""
    assert write_incremental_only(False) is False

    enabled_token = incremental_checkpoints_enabled.set(False)
    try:
        assert write_incremental_only(True) is True
    finally:
        incremental_checkpoints_enabled.reset(enabled_token)


def test_compaction_ui_chat_log_update_appends_without_incremental_only():
    entries = [{"message_id": "m1"}]

    assert compaction_ui_chat_log_update(entries) is entries


@pytest.mark.usefixtures("incremental_checkpoints_only")
def test_compaction_ui_chat_log_update_replaces_channel_under_incremental_only():
    entries = [{"message_id": "m1"}]

    update = compaction_ui_chat_log_update(entries)

    assert isinstance(update, Overwrite)
    assert update.value is entries
