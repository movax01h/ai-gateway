from unittest.mock import Mock

import pytest
from google.protobuf.struct_pb2 import Struct
from pydantic import ValidationError

from contract import contract_pb2
from duo_workflow_service.agent_platform.utils.flow import VALID_SCHEMA_VERSIONS
from duo_workflow_service.flow_request import (
    _LEGACY_WORKFLOW_NAMES,
    InlineFlowRequest,
    LegacyWorkflowRequest,
    RegistryFlowRequest,
    normalize_flow_request,
    workflow_definition_key_from_proto,
)
from duo_workflow_service.workflows.registry import (
    _FLOW_BY_VERSIONS,
    _FLOW_CONFIGS_BY_VERSION,
    _WORKFLOWS_LOOKUP,
)
from lib.events import GLReportingEventContext


class TestRegistryFlowRequest:
    def test_valid_v1(self):
        req = RegistryFlowRequest(
            config_id="developer", schema_version="v1", version="1.0.0"
        )
        assert req.to_legacy_identifier() == "developer/v1"

    def test_valid_experimental(self):
        req = RegistryFlowRequest(
            config_id="duo_planner", schema_version="experimental", version="2.0.0"
        )
        assert req.to_legacy_identifier() == "duo_planner/experimental"

    def test_invalid_schema_version(self):
        with pytest.raises(ValidationError, match="flowConfigSchemaVersion"):
            RegistryFlowRequest(
                config_id="developer", schema_version="v99", version="1.0.0"
            )

    def test_from_legacy_definition(self):
        req = RegistryFlowRequest.from_legacy_definition("developer/v1")
        assert req.config_id == "developer"
        assert req.schema_version == "v1"
        assert req.version == "1.0.0"

    def test_from_legacy_definition_experimental(self):
        req = RegistryFlowRequest.from_legacy_definition("duo_planner/experimental")
        assert req.config_id == "duo_planner"
        assert req.schema_version == "experimental"
        assert req.version == "1.0.0"

    def test_from_legacy_definition_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid workflow_definition format"):
            RegistryFlowRequest.from_legacy_definition("no_slash_here")

    def test_from_legacy_definition_invalid_api_version(self):
        with pytest.raises(ValueError, match="Invalid API version"):
            RegistryFlowRequest.from_legacy_definition("developer/v99")


class TestNormalizeFlowRequest:
    def test_structured_fields_return_registry_request(self):
        req = contract_pb2.StartWorkflowRequest(
            flowConfigId="developer",
            flowConfigSchemaVersion="v1",
            flowVersion="2.0.0",
        )
        result = normalize_flow_request(req, lsp_version=Mock())
        assert isinstance(result, RegistryFlowRequest)
        assert result.config_id == "developer"
        assert result.schema_version == "v1"
        assert result.version == "2.0.0"

    def test_structured_fields_missing_schema_version(self):
        req = contract_pb2.StartWorkflowRequest(
            flowConfigId="developer",
            flowVersion="1.0.0",
        )
        with pytest.raises(ValueError, match="flowConfigSchemaVersion"):
            normalize_flow_request(req, lsp_version=Mock())

    def test_structured_fields_missing_flow_version(self):
        req = contract_pb2.StartWorkflowRequest(
            flowConfigId="developer",
            flowConfigSchemaVersion="v1",
        )
        with pytest.raises(ValueError, match="flowVersion"):
            normalize_flow_request(req, lsp_version=Mock())

    def test_structured_fields_conflict_with_flow_config(self):
        flow_config = Struct()
        flow_config.update({"version": "v1", "environment": "test"})
        req = contract_pb2.StartWorkflowRequest(
            flowConfigId="developer",
            flowConfigSchemaVersion="v1",
            flowVersion="1.0.0",
            flowConfig=flow_config,
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            normalize_flow_request(req, lsp_version=Mock())


class TestNormalizeFlowInlineConfig:
    def test_inline_config_returns_inline_request(self):
        flow_config = Struct()
        flow_config.update({"version": "v1", "environment": "test"})
        req = contract_pb2.StartWorkflowRequest(
            flowConfig=flow_config,
            flowConfigSchemaVersion="v1",
        )
        result = normalize_flow_request(req, lsp_version=Mock())
        assert isinstance(result, InlineFlowRequest)
        assert result.schema_version == "v1"

    def test_inline_config_lsp_override_extracts_version_from_struct(self):
        flow_config = Struct()
        flow_config.update({"version": "experimental", "environment": "test"})
        req = contract_pb2.StartWorkflowRequest(
            flowConfig=flow_config,
        )
        # No lsp_version → triggers the override
        result = normalize_flow_request(req, lsp_version=None)
        assert isinstance(result, InlineFlowRequest)
        assert result.schema_version == "experimental"

    def test_inline_config_missing_schema_version_no_lsp(self):
        flow_config = Struct()
        flow_config.update({"environment": "test"})  # no "version" key
        req = contract_pb2.StartWorkflowRequest(
            flowConfig=flow_config,
        )
        with pytest.raises(ValueError, match="flowConfigSchemaVersion"):
            normalize_flow_request(req, lsp_version=None)


class TestNormalizeFlowLegacyWorkflowDefinition:
    def test_legacy_string_returns_registry_request(self):
        req = contract_pb2.StartWorkflowRequest(
            workflowDefinition="developer/v1",
        )
        result = normalize_flow_request(req, lsp_version=Mock())
        assert isinstance(result, RegistryFlowRequest)
        assert result.config_id == "developer"
        assert result.schema_version == "v1"
        assert result.version == "1.0.0"

    def test_legacy_named_workflow_returns_legacy_request(self):
        req = contract_pb2.StartWorkflowRequest(
            workflowDefinition="software_development",
        )
        result = normalize_flow_request(req, lsp_version=Mock())
        assert isinstance(result, LegacyWorkflowRequest)
        assert result.workflow_definition == "software_development"

    def test_legacy_named_chat(self):
        req = contract_pb2.StartWorkflowRequest(
            workflowDefinition="chat",
        )
        result = normalize_flow_request(req, lsp_version=Mock())
        assert isinstance(result, LegacyWorkflowRequest)
        assert result.workflow_definition == "chat"


class TestNormalizeFlowEdgeCases:
    def test_empty_request_defaults_to_software_development(self):
        req = contract_pb2.StartWorkflowRequest()
        result = normalize_flow_request(req, lsp_version=Mock())
        assert isinstance(result, LegacyWorkflowRequest)
        assert result.workflow_definition == "software_development"

    def test_flow_version_without_flow_config_id_raises(self):
        req = contract_pb2.StartWorkflowRequest(
            flowVersion="2.0.0",
        )
        with pytest.raises(ValueError, match="flowVersion requires flowConfigId"):
            normalize_flow_request(req, lsp_version=Mock())

    def test_structured_fields_take_priority_over_workflow_definition(self):
        req = contract_pb2.StartWorkflowRequest(
            flowConfigId="developer",
            flowConfigSchemaVersion="v1",
            flowVersion="1.0.0",
            workflowDefinition="software_development",
        )
        result = normalize_flow_request(req, lsp_version=Mock())
        assert isinstance(result, RegistryFlowRequest)
        assert result.config_id == "developer"

    def test_empty_flow_config_id_raises(self):
        req = contract_pb2.StartWorkflowRequest(
            flowConfigId="",
            flowConfigSchemaVersion="v1",
            flowVersion="1.0.0",
        )
        with pytest.raises(ValidationError, match="flowConfigId cannot be empty"):
            normalize_flow_request(req, lsp_version=Mock())

    def test_empty_flow_version_raises(self):
        req = contract_pb2.StartWorkflowRequest(
            flowConfigId="developer",
            flowConfigSchemaVersion="v1",
            flowVersion="",
        )
        with pytest.raises(ValidationError, match="flowVersion cannot be empty"):
            normalize_flow_request(req, lsp_version=Mock())

    def test_inline_config_ignores_legacy_workflow_definition(self):
        """Inline config takes priority over legacy workflowDefinition."""
        flow_config = Struct()
        flow_config.update({"version": "v1", "environment": "test"})
        req = contract_pb2.StartWorkflowRequest(
            flowConfig=flow_config,
            flowConfigSchemaVersion="v1",
            workflowDefinition="chat",
        )
        result = normalize_flow_request(req, lsp_version=Mock())
        assert isinstance(result, InlineFlowRequest)
        assert result.schema_version == "v1"


class TestWorkflowDefinitionKeyFromProto:
    """Tests for the lightweight interceptor helper."""

    def test_structured_fields_take_priority(self):
        req = contract_pb2.StartWorkflowRequest(
            flowConfigId="developer",
            flowConfigSchemaVersion="v1",
            workflowDefinition="software_development",
        )
        assert workflow_definition_key_from_proto(req) == "developer/v1"

    def test_structured_fields_without_legacy(self):
        req = contract_pb2.StartWorkflowRequest(
            flowConfigId="developer",
            flowConfigSchemaVersion="v1",
        )
        assert workflow_definition_key_from_proto(req) == "developer/v1"

    def test_flow_version_excluded_from_key(self):
        req = contract_pb2.StartWorkflowRequest(
            flowConfigId="developer",
            flowConfigSchemaVersion="v1",
            flowVersion="2.0.0",
        )
        assert workflow_definition_key_from_proto(req) == "developer/v1"

    def test_legacy_definition_when_no_structured_fields(self):
        req = contract_pb2.StartWorkflowRequest(
            workflowDefinition="software_development",
        )
        assert workflow_definition_key_from_proto(req) == "software_development"

    def test_empty_request(self):
        req = contract_pb2.StartWorkflowRequest()
        assert workflow_definition_key_from_proto(req) == ""

    def test_flow_config_id_without_schema_version_falls_back(self):
        req = contract_pb2.StartWorkflowRequest(
            flowConfigId="developer",
            workflowDefinition="software_development",
        )
        assert workflow_definition_key_from_proto(req) == "software_development"

    def test_experimental(self):
        req = contract_pb2.StartWorkflowRequest(
            flowConfigId="duo_planner",
            flowConfigSchemaVersion="experimental",
        )
        assert workflow_definition_key_from_proto(req) == "duo_planner/experimental"

    def test_inline_flow_config_with_workflow_definition_returns_workflow_definition(
        self,
    ):
        """On main, billing always receives request.workflowDefinition verbatim.

        When a client sends flowConfig alongside workflowDefinition, billing must see the workflowDefinition — not the
        schema version.
        """
        flow_config = Struct()
        flow_config.update({"version": "v1", "environment": "test"})
        req = contract_pb2.StartWorkflowRequest(
            flowConfig=flow_config,
            flowConfigSchemaVersion="v1",
            workflowDefinition="software_development",
        )
        assert workflow_definition_key_from_proto(req) == "software_development"

    def test_inline_flow_config_without_workflow_definition_returns_empty(self):
        """workflow_definition_key_from_proto just extracts — it does not default.

        The downstream from_workflow_definition handles "" → "software_development".
        """
        flow_config = Struct()
        flow_config.update({"version": "v1", "environment": "test"})
        req = contract_pb2.StartWorkflowRequest(
            flowConfig=flow_config,
            flowConfigSchemaVersion="v1",
        )
        assert workflow_definition_key_from_proto(req) == ""


class TestBillingContextFromProto:
    """Integration tests: proto → workflow_definition_key_from_proto → GLReportingEventContext.

    These replicate the exact chain in the usage_quota interceptor to ensure
    billing sees the correct values.
    """

    @staticmethod
    def _billing_context(req):
        """Simulate the usage_quota interceptor chain on main."""
        key = workflow_definition_key_from_proto(req)
        return GLReportingEventContext.from_workflow_definition(
            key,
            is_ai_catalog_item=bool(req.flowConfig),
        )

    def test_legacy_workflow(self):
        req = contract_pb2.StartWorkflowRequest(
            workflowDefinition="software_development",
        )
        ctx = self._billing_context(req)
        assert ctx.feature_qualified_name == "software_development"
        assert ctx.value == "software_development"
        assert ctx.feature_ai_catalog_item is False

    def test_registry_flow(self):
        req = contract_pb2.StartWorkflowRequest(
            flowConfigId="developer",
            flowConfigSchemaVersion="v1",
            flowVersion="1.0.0",
        )
        ctx = self._billing_context(req)
        assert ctx.feature_qualified_name == "developer/v1"
        assert ctx.value == "developer"
        assert ctx.feature_ai_catalog_item is False

    def test_inline_flow_config_with_workflow_definition(self):
        """Client sends flowConfig + workflowDefinition — billing sees workflowDefinition."""
        flow_config = Struct()
        flow_config.update({"version": "v1"})
        req = contract_pb2.StartWorkflowRequest(
            flowConfig=flow_config,
            flowConfigSchemaVersion="v1",
            workflowDefinition="software_development",
        )
        ctx = self._billing_context(req)
        assert ctx.feature_qualified_name == "software_development"
        assert ctx.value == "software_development"
        assert ctx.feature_ai_catalog_item is True

    def test_inline_flow_config_without_workflow_definition(self):
        """Client sends flowConfig without workflowDefinition — billing defaults to software_development."""
        flow_config = Struct()
        flow_config.update({"version": "v1"})
        req = contract_pb2.StartWorkflowRequest(
            flowConfig=flow_config,
            flowConfigSchemaVersion="v1",
        )
        ctx = self._billing_context(req)
        assert ctx.feature_qualified_name == "software_development"
        assert ctx.value == "software_development"
        assert ctx.feature_ai_catalog_item is True

    def test_empty_request_defaults_to_software_development(self):
        req = contract_pb2.StartWorkflowRequest()
        ctx = self._billing_context(req)
        assert ctx.feature_qualified_name == "software_development"
        assert ctx.value == "software_development"


def test_legacy_workflow_names_match_workflows_lookup():
    """Guard against _LEGACY_WORKFLOW_NAMES drifting from _WORKFLOWS_LOOKUP."""
    assert _LEGACY_WORKFLOW_NAMES == set(_WORKFLOWS_LOOKUP.keys())


def test_valid_schema_versions_match_flow_registries():
    """Guard against VALID_SCHEMA_VERSIONS drifting from the resolution registries."""
    assert VALID_SCHEMA_VERSIONS == set(_FLOW_BY_VERSIONS.keys())
    assert VALID_SCHEMA_VERSIONS == set(_FLOW_CONFIGS_BY_VERSION.keys())
