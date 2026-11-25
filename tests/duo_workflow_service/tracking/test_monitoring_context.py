from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from duo_workflow_service.tracking import MonitoringContext


class TestFlowVersioningFields:
    def test_returns_all_fields_when_set(self):
        context = MonitoringContext(
            flow_id="developer",
            flow_version="1.2.3",
            schema_version="v1",
        )
        assert context.flow_versioning_fields() == {
            "flow_id": "developer",
            "flow_version": "1.2.3",
            "schema_version": "v1",
        }

    def test_omits_unset_fields(self):
        # Inline flows only carry a schema version.
        context = MonitoringContext(schema_version="v1")
        assert context.flow_versioning_fields() == {"schema_version": "v1"}

    def test_returns_empty_dict_for_legacy_flows(self):
        # Legacy flows carry no versioning information.
        context = MonitoringContext()
        assert not context.flow_versioning_fields()

    def test_omits_empty_string_fields(self):
        # Empty strings are treated as unset — they should not leak into traces.
        context = MonitoringContext(flow_id="", flow_version="", schema_version="")
        assert context.flow_versioning_fields() == {}


class TestSetFlowIdentity:
    def test_sets_all_fields(self):
        context = MonitoringContext()
        context.set_flow_identity(
            flow_id="developer",
            flow_version="1.2.3",
            schema_version="v1",
        )
        assert context.flow_id == "developer"
        assert context.flow_version == "1.2.3"
        assert context.schema_version == "v1"

    def test_sets_partial_fields(self):
        context = MonitoringContext()
        context.set_flow_identity(schema_version="v1")
        assert context.flow_id is None
        assert context.flow_version is None
        assert context.schema_version == "v1"

    def test_ignores_empty_strings(self):
        context = MonitoringContext(flow_id="original")
        context.set_flow_identity(flow_id="", flow_version="")
        # Empty strings are not stored; existing values are preserved.
        assert context.flow_id == "original"
        assert context.flow_version is None

    def test_ignores_none_values(self):
        context = MonitoringContext(flow_id="original")
        context.set_flow_identity(flow_id=None)
        assert context.flow_id == "original"

    def test_rejects_unknown_keys(self):
        # The ** unpacking contract: unknown keys from tracking_fields() cause a TypeError.
        context = MonitoringContext()
        kwargs: dict[str, str] = dict(zip(["flow_id", "bogus_field"], ["ok", "bad"]))
        with pytest.raises(TypeError, match="bogus_field"):
            context.set_flow_identity(**kwargs)

    def test_no_op_with_empty_dict(self):
        # Legacy flows return {} from tracking_fields(); set_flow_identity(**{}) is a no-op.
        context = MonitoringContext()
        context.set_flow_identity()
        assert context.flow_id is None
        assert context.flow_version is None
        assert context.schema_version is None


class TestSetModelIdentifier:
    @pytest.mark.parametrize(
        ("model_metadata", "expected"),
        [
            # GitLab models carry the GitLab identifier in `name`.
            (
                SimpleNamespace(
                    name="claude_sonnet_5_vertex", identifier=None, api_key="secret"
                ),
                "claude_sonnet_5_vertex",
            ),
            # Custom and self-hosted models carry the model in `identifier`.
            (
                SimpleNamespace(
                    name="claude_3", identifier="bedrock/claude-3", api_key="secret"
                ),
                "bedrock/claude-3",
            ),
            # Values that are not strings are ignored.
            (MagicMock(), None),
            (SimpleNamespace(name="", identifier=""), None),
            (None, None),
        ],
    )
    def test_sets_model_identifier(self, model_metadata, expected):
        context = MonitoringContext()
        context.set_model_identifier(model_metadata)
        assert context.model_identifier == expected
        assert "secret" not in context.model_dump_json()

    def test_keeps_existing_value_when_nothing_resolves(self):
        context = MonitoringContext(model_identifier="claude_sonnet_5_vertex")
        context.set_model_identifier(None)
        assert context.model_identifier == "claude_sonnet_5_vertex"
