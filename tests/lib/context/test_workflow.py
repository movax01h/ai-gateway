"""Tests for lib/context/workflow module."""

import pytest

from lib.context.workflow import _workflow_id, get_workflow_id, set_workflow_id


@pytest.fixture(autouse=True)
def reset_workflow_id():
    token = _workflow_id.set("undefined")
    yield
    _workflow_id.reset(token)


def test_set_and_get_workflow_id():
    """set_workflow_id stores the value; get_workflow_id returns it."""
    set_workflow_id("wf-abc-123")
    assert get_workflow_id() == "wf-abc-123"


def test_get_workflow_id_returns_none_for_default():
    """get_workflow_id returns None when the ContextVar holds the sentinel 'undefined'."""
    assert get_workflow_id() is None
