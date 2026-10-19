from urllib.parse import parse_qs, urlparse

import pytest
from langchain_core.messages import AIMessage

from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    add_compression_param,
    compress_checkpoint,
    uncompress_checkpoint,
)


@pytest.mark.parametrize(
    "endpoint,expected_params",
    [
        (
            "/api/v4/ai/duo_workflows/workflows/1/checkpoints",
            {"accept_compressed": ["true"]},
        ),
        (
            "/api/v4/ai/duo_workflows/workflows/1/checkpoints?per_page=1",
            {"accept_compressed": ["true"], "per_page": ["1"]},
        ),
        # A blank `checkpoint_ns` means "the flow's own top-level checkpoint lineage", which
        # is not the same as omitting the parameter (the list endpoint's unfiltered "every
        # lineage" default), so it must survive the round-trip.
        (
            "/api/v4/ai/duo_workflows/workflows/1/checkpoints?per_page=1&checkpoint_ns=",
            {
                "accept_compressed": ["true"],
                "per_page": ["1"],
                "checkpoint_ns": [""],
            },
        ),
        (
            "/api/v4/ai/duo_workflows/workflows/1/checkpoints?checkpoint_ns=delegation%3Atask-1",
            {
                "accept_compressed": ["true"],
                "checkpoint_ns": ["delegation:task-1"],
            },
        ),
    ],
    ids=["no_query", "existing_param", "blank_checkpoint_ns", "nested_checkpoint_ns"],
)
def test_add_compression_param_preserves_existing_query(endpoint, expected_params):
    parsed = urlparse(add_compression_param(endpoint))

    assert parsed.path == "/api/v4/ai/duo_workflows/workflows/1/checkpoints"
    assert parse_qs(parsed.query, keep_blank_values=True) == expected_params


def test_uncompress_checkpoint_decodes_messages_by_default():
    compressed = compress_checkpoint({"channel_values": {"m": AIMessage(content="hi")}})

    message = uncompress_checkpoint(compressed)["channel_values"]["m"]

    assert isinstance(message, AIMessage)
    assert message.content == "hi"


def test_uncompress_checkpoint_without_object_hook_keeps_dicts():
    compressed = compress_checkpoint({"channel_values": {"m": AIMessage(content="hi")}})

    message = uncompress_checkpoint(compressed, object_hook=None)["channel_values"]["m"]

    assert message["type"] == "AIMessage"
    assert message["content"] == "hi"
