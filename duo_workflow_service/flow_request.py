"""Typed flow request models and normalization.

All incoming StartWorkflowRequest proto messages are normalized into one of three FlowRequest types at the boundary.
Downstream code works only with these types — it never re-examines raw proto fields.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Union

from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from pydantic import BaseModel, ConfigDict, model_validator

from contract import contract_pb2
from duo_workflow_service.agent_platform.utils.flow import (
    VALID_SCHEMA_VERSIONS,
    parse_deprecated_workflow_definition,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    DEFAULT_FLOW_VERSION,
)
from lib.language_server import LanguageServerVersion

_LEGACY_WORKFLOW_NAMES = frozenset(
    {
        "software_development",
        "convert_to_gitlab_ci",
        "chat",
        "issue_to_merge_request",
    }
)


class BaseFlowRequest(BaseModel):
    """Base class for all flow request types."""

    @abstractmethod
    def to_legacy_identifier(self) -> str:
        """Return the deprecated ``workflowDefinition``-style string.

        Downstream consumers (GLReportingEventContext, billing, monitoring, logging) still expect this format.  A
        follow-up should migrate those consumers to accept FlowRequest directly, at which point this method is deleted.
        """


class RegistryFlowRequest(BaseFlowRequest):
    """Resolve a flow from the YAML registry by name + version."""

    config_id: str
    schema_version: str
    version: str

    @model_validator(mode="after")
    def _validate_fields(self) -> RegistryFlowRequest:
        if not self.config_id:
            raise ValueError("flowConfigId cannot be empty.")
        if not self.version:
            raise ValueError("flowVersion cannot be empty.")
        if self.schema_version not in VALID_SCHEMA_VERSIONS:
            raise ValueError(
                f"Invalid flowConfigSchemaVersion: '{self.schema_version}'. "
                f"Must be one of: {', '.join(sorted(VALID_SCHEMA_VERSIONS))}."
            )
        return self

    @classmethod
    def from_legacy_definition(cls, workflow_definition: str) -> RegistryFlowRequest:
        """Parse legacy workflow definitions like 'developer/v1' into a registry request with version '1.0.0'.

        Legacy clients don't have flow versioning, so we pin to 1.0.0.
        """
        api_version, flow_name = parse_deprecated_workflow_definition(
            workflow_definition
        )
        return cls(
            config_id=flow_name,
            schema_version=api_version,
            version=DEFAULT_FLOW_VERSION,
        )

    def to_legacy_identifier(self) -> str:
        return f"{self.config_id}/{self.schema_version}"


class InlineFlowRequest(BaseFlowRequest):
    """Resolve a flow from an inline protobuf Struct config."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config_struct: Struct
    schema_version: str
    workflow_definition: str = ""

    def to_legacy_identifier(self) -> str:
        return self.workflow_definition


class LegacyWorkflowRequest(BaseFlowRequest):
    """Pre-registry Python workflow class (software_development, chat, etc.)."""

    workflow_definition: str

    def to_legacy_identifier(self) -> str:
        return self.workflow_definition


FlowRequest = Union[RegistryFlowRequest, InlineFlowRequest, LegacyWorkflowRequest]


def normalize_flow_request(
    start_req: contract_pb2.StartWorkflowRequest,
    lsp_version: Optional[LanguageServerVersion],
) -> FlowRequest:
    """Translate raw proto fields into a validated FlowRequest.

    All validation and legacy translation happens here.
    Downstream code never touches raw proto fields for flow resolution.

    Raises:
        ValueError: On invalid or conflicting field combinations.
    """
    has_flow_config_id = start_req.HasField("flowConfigId")
    has_flow_version = start_req.HasField("flowVersion")
    has_flow_config = start_req.HasField("flowConfig")
    flow_config_schema_version = start_req.flowConfigSchemaVersion or None

    # ── Path A: new structured fields ──
    if has_flow_config_id:
        if has_flow_config:
            raise ValueError(
                "flowConfigId and flowConfig are mutually exclusive — "
                "set one or the other, not both."
            )
        if not flow_config_schema_version:
            raise ValueError(
                "flowConfigId requires flowConfigSchemaVersion to also be provided."
            )
        if not has_flow_version:
            raise ValueError("flowConfigId requires flowVersion to also be provided.")
        return RegistryFlowRequest(
            config_id=start_req.flowConfigId,
            schema_version=flow_config_schema_version,
            version=start_req.flowVersion,
        )

    # ── Path B: inline config struct ──
    if has_flow_config:
        # LSP override: old clients embed the version inside the struct
        if not lsp_version or lsp_version.ignore_broken_flow_schema_version():
            flow_config_schema_version = MessageToDict(start_req.flowConfig).get(
                "version"
            )
        if not flow_config_schema_version:
            raise ValueError(
                "flowConfig requires flowConfigSchemaVersion to also be provided."
            )
        return InlineFlowRequest(
            config_struct=start_req.flowConfig,
            schema_version=flow_config_schema_version,
            workflow_definition=start_req.workflowDefinition or "",
        )

    # ── Path C: legacy workflowDefinition string ──
    workflow_definition = start_req.workflowDefinition or None
    if has_flow_version:
        raise ValueError("flowVersion requires flowConfigId to also be provided.")

    if workflow_definition:
        if workflow_definition in _LEGACY_WORKFLOW_NAMES:
            return LegacyWorkflowRequest(workflow_definition=workflow_definition)
        return RegistryFlowRequest.from_legacy_definition(workflow_definition)

    # ── Path D: empty request → default ──
    return LegacyWorkflowRequest(workflow_definition="software_development")


def workflow_definition_key_from_proto(
    request: contract_pb2.StartWorkflowRequest,
) -> str:
    """Lightweight extraction for interceptors that run before the server handler.

    Returns the deprecated workflowDefinition-style string for auth, billing, and metrics.  flowVersion is intentionally
    excluded — billing keys on flow identity, not patch version.
    """
    if request.HasField("flowConfigId") and request.flowConfigSchemaVersion:
        return f"{request.flowConfigId}/{request.flowConfigSchemaVersion}"
    return request.workflowDefinition
