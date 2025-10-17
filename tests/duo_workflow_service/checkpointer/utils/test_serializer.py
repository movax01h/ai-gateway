import ormsgpack
import pytest
from langgraph.types import Interrupt

from duo_workflow_service.checkpointer.utils.serializer import (
    CheckpointSerializer,
    CheckpointValidationError,
)


@pytest.fixture
def serializer():
    return CheckpointSerializer()


@pytest.mark.parametrize(
    "value,resumable,ns,when",
    [
        ("Simple workflow", True, ["node1"], "during"),
        ("Multi-node workflow", False, ["node1", "node2", "node3"], "after"),
        ("Unicode: 你好 🎉", True, ["planning"], "during"),
        ("", True, [], "during"),
    ],
)
def test_interrupt_roundtrip(serializer, value, resumable, ns, when):
    """Interrupt objects serialize and deserialize correctly."""
    interrupt = Interrupt(value=value, resumable=resumable, ns=ns, when=when)

    type_str, data = serializer.dumps_typed(interrupt)
    result = serializer.loads_typed((type_str, data))

    assert type_str == "msgpack"
    assert isinstance(result, Interrupt)
    assert result.value == value
    assert result.resumable == resumable
    assert result.ns == ns
    assert result.when == when


def test_large_checkpoint_data(serializer):
    """Large checkpoint payloads are handled efficiently."""
    interrupt = Interrupt(
        value="x" * 10000,
        resumable=True,
        ns=[f"node_{i}" for i in range(100)],
        when="during",
    )

    type_str, data = serializer.dumps_typed(interrupt)
    result = serializer.loads_typed((type_str, data))

    assert len(result.value) == 10000
    assert len(result.ns) == 100


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2, 3],
        {"key": "value"},
        "simple string",
    ],
)
def test_simple_types_roundtrip(serializer, data):
    """Simple data types serialize correctly."""
    msgpack_data = ormsgpack.packb(data)
    result = serializer.loads_typed(("msgpack", msgpack_data))

    assert result == data


@pytest.mark.parametrize("format_type", ["pickle", "json", "yaml", "xml"])
def test_unsupported_formats_rejected(serializer, format_type):
    """Only msgpack format is supported for consistency."""
    with pytest.raises(CheckpointValidationError) as exc:
        serializer.loads_typed((format_type, b"data"))

    assert "unsupported" in str(exc.value).lower()


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("workflow.legacy", "OldWorkflowClass"),
        ("custom.extension", "CustomHandler"),
        ("third_party.plugin", "PluginClass"),
    ],
)
def test_non_standard_modules_rejected(serializer, module_name, class_name):
    """Checkpoint data must use standard langgraph modules."""
    from langgraph.checkpoint.serde.jsonplus import _msgpack_enc

    ext = ormsgpack.Ext(2, _msgpack_enc((module_name, class_name, {})))
    msgpack_data = ormsgpack.packb(ext)

    with pytest.raises(CheckpointValidationError):
        serializer.loads_typed(("msgpack", msgpack_data))


def test_unknown_class_handled_gracefully(serializer):
    """References to unknown classes fail safely."""
    from langgraph.checkpoint.serde.jsonplus import _msgpack_enc

    ext = ormsgpack.Ext(2, _msgpack_enc(("langgraph.types", "UnknownClass", {})))
    msgpack_data = ormsgpack.packb(ext)

    result = serializer.loads_typed(("msgpack", msgpack_data))

    assert result is None
