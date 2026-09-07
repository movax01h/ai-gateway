"""Typed flow request models and normalization.

All incoming StartWorkflowRequest proto messages are normalized into one of three FlowRequest types at the boundary.
Downstream code works only with these types — it never re-examines raw proto fields.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Union, override

from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from pydantic import BaseModel, ConfigDict, model_validator

from contract import contract_pb2
from duo_workflow_service.agent_platform.utils.flow import (
    VALID_SCHEMA_VERSIONS,
    parse_deprecated_workflow_definition,
)
from duo_workflow_service.agent_platform.v1.catalog import (
    CatalogItems,
    CatalogItemsError,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    DEFAULT_FLOW_VERSION,
)
from lib.language_server import LanguageServerVersion

# Only the v1 flow config declares an `include` section, and only the v1 Flow accepts
# the items argument.
_CATALOG_SCHEMA_VERSION = "v1"

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

    def supports_catalog_items(self) -> bool:
        """Whether catalog items sent with this request could bind to anything.

        Returns:
            ``False`` by default; only the request types whose flows accept items override it.
        """
        return False


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

    @override
    def supports_catalog_items(self) -> bool:
        return self.schema_version == _CATALOG_SCHEMA_VERSION


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
    # Treat empty strings as "not provided". Some clients (e.g. duo-cli 8.92.0) explicitly
    # set these fields to "" in the proto, which makes HasField() return True even though
    # semantically the client meant "unset". Without this guard, Path A activates with an
    # empty config_id and the validator rejects the request.
    has_flow_config_id = start_req.HasField("flowConfigId") and bool(
        start_req.flowConfigId
    )
    has_flow_version = start_req.HasField("flowVersion") and bool(start_req.flowVersion)
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


def normalize_catalog_items(
    start_req: contract_pb2.StartWorkflowRequest,
    flow_request: FlowRequest,
) -> CatalogItems:
    """Translate the ``catalog_items`` envelope into validated items.

    Args:
        start_req: The request as received.
        flow_request: The already-normalized flow identity, which decides whether items can bind.

    Returns:
        The items the request carried, or none when it carried no envelope, the common case.

    Raises:
        CatalogItemsError: If this request cannot carry items, or if the ones it carries are not
            valid.
    """
    if not start_req.HasField("catalog_items"):
        return CatalogItems()

    # Rejected rather than dropped, so a client never sees a flow that quietly declines
    # to delegate.
    if not flow_request.supports_catalog_items():
        raise CatalogItemsError(
            f"Catalog items are only supported for '{_CATALOG_SCHEMA_VERSION}' flows "
            "shipped as foundational flows."
        )

    version = start_req.catalog_items.WhichOneof("items")
    if version is None:
        # A client built against a newer schema. It intended to send items, so failing is
        # more honest than running without them.
        raise CatalogItemsError(
            "The catalog_items envelope carries no schema version this server recognises."
        )

    # Proto field names are kept, so the payload keys match the item models.
    return CatalogItems.from_payload(
        MessageToDict(
            getattr(start_req.catalog_items, version),
            preserving_proto_field_name=True,
        )
    )


def workflow_definition_key_from_proto(
    request: contract_pb2.StartWorkflowRequest,
) -> str:
    """Lightweight extraction for interceptors that run before the server handler.

    Returns the deprecated workflowDefinition-style string for auth, billing, and metrics.  flowVersion is intentionally
    excluded — billing keys on flow identity, not patch version.
    """
    # Guard against clients that explicitly set flowConfigId="" (e.g. duo-cli 8.92.0, Rails with nil
    # flow_config_id). HasField() returns True even for empty strings, so without this check the
    # billing key becomes "/v1" instead of falling back to workflowDefinition.
    # Mirrors the same guard in normalize_flow_request.
    if (
        request.HasField("flowConfigId")
        and request.flowConfigId
        and request.flowConfigSchemaVersion
    ):
        return f"{request.flowConfigId}/{request.flowConfigSchemaVersion}"
    return request.workflowDefinition
