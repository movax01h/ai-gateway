"""Tests for the block_pickle module."""

import io
import pickle
import sys

import pytest

# pylint: disable=import-outside-toplevel,unused-import
# inline imports are necessary to avoid spilling monkeypatch over unrelated tests


def test_pickle_load_is_blocked():
    """Test that pickle.load is blocked."""
    from duo_workflow_service.block_pickle import PickleDisabledError

    with pytest.raises(PickleDisabledError, match="Unpickling is disabled"):
        pickle.load(None)


def test_pickle_loads_is_blocked():
    """Test that pickle.loads is blocked."""
    from duo_workflow_service.block_pickle import PickleDisabledError

    with pytest.raises(PickleDisabledError, match="Unpickling is disabled"):
        pickle.loads(b"test")


def test_pickle_unpickler_is_blocked():
    """Test that pickle.Unpickler is blocked."""
    from duo_workflow_service.block_pickle import PickleDisabledError

    with pytest.raises(PickleDisabledError, match="pickle.Unpickler is disabled"):
        pickle.Unpickler(None)


def test_pickletools_import_is_blocked():
    """Test that importing pickletools is blocked."""
    from duo_workflow_service.block_pickle import PickleDisabledError

    # Remove pickletools from sys.modules if it's already imported
    if "pickletools" in sys.modules:
        del sys.modules["pickletools"]

    with pytest.raises(PickleDisabledError, match="Import blocked: pickletools"):
        import pickletools


def test_pickle_dump_still_works():
    """Test that pickle.dump still works (only unpickling is blocked)."""
    import duo_workflow_service.block_pickle

    data = {"key": "value"}
    buffer = io.BytesIO()

    # This should not raise an error
    pickle.dump(data, buffer)

    # Verify data was written
    assert buffer.getvalue() != b""


def test_pickle_dumps_still_works():
    """Test that pickle.dumps still works (only unpickling is blocked)."""
    import duo_workflow_service.block_pickle

    data = {"key": "value"}

    # This should not raise an error
    result = pickle.dumps(data)

    # Verify data was serialized
    assert result != b""
