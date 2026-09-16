from urllib.parse import parse_qs, urlparse

import pytest

from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    add_compression_param,
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
