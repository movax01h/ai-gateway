from typing import Any, Literal, Optional, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from duo_workflow_service.tools.ascp.queries import (
    CREATE_ASCP_SECURITY_CONTEXT_MUTATION,
)
from duo_workflow_service.tools.ascp.types import AscpSeverityLiteral
from duo_workflow_service.tools.ascp.utils import parse_graphql_errors
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class AscpSecurityGuidelineInput(BaseModel):
    """Input model for a single security guideline."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    name: str = Field(
        description="Name of the security guideline.",
    )
    operation: str = Field(
        description="Operation this guideline applies to (e.g., 'READ', 'WRITE', 'DELETE').",
    )
    legitimate_use: Optional[str] = Field(
        default=None,
        description="Description of legitimate use for this operation.",
    )
    security_boundary: Optional[str] = Field(
        default=None,
        description="Security boundary conditions for this guideline.",
    )
    business_context: Optional[str] = Field(
        default=None,
        description="Business context explaining why this guideline exists.",
    )
    severity_if_violated: Optional[AscpSeverityLiteral] = Field(
        default="MEDIUM",
        description=(
            'Severity level if this guideline is violated: "LOW", "MEDIUM", "HIGH",'
            ' or "CRITICAL". Defaults to "MEDIUM".'
        ),
    )


class CreateAscpSecurityContextResponse(BaseModel):
    """Response model for creating an ASCP security context."""

    security_context: dict[str, Any]


class CreateAscpSecurityContextInput(BaseModel):
    """Input model for the CreateAscpSecurityContext tool."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    project_path: str = Field(
        description='Full path of the project (e.g., "namespace/project").',
    )
    component_id: str = Field(
        description=(
            "GraphQL ID of the ASCP component to attach this security context to"
            ' (e.g., "gid://gitlab/Ascp::Component/1").'
        ),
    )
    scan_id: str = Field(
        description='GraphQL ID of the ASCP scan this security context belongs to (e.g., "gid://gitlab/Ascp::Scan/1").',
    )
    guidelines: list[AscpSecurityGuidelineInput] = Field(
        min_length=1,
        description="List of security guidelines for this context. At least one is required.",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Optional summary of the security context.",
    )
    authentication_model: Optional[Literal["true", "false"]] = Field(
        default=None,
        description="Whether the component requires authentication.",
    )
    authorization_model: Optional[Literal["elevated", "standard"]] = Field(
        default=None,
        description="Whether the component requires elevated privileges.",
    )
    data_sensitivity: Optional[Literal["true", "false"]] = Field(
        default=None,
        description=(
            "Whether the component handles sensitive data "
            "(PII, credentials, secrets, tokens)."
        ),
    )


class CreateAscpSecurityContext(DuoBaseTool):
    """Tool for creating an ASCP (Application Security Collaboration Platform) security context.

    On success, returns JSON with the created security context details. On error, raises ToolException with error
    details.
    """

    name: str = "ascp_create_security_context"
    description: str = """
    Create a new ASCP security context for a component within a given scan.

    Use this tool when you need to define the security context for an ASCP component,
    including security guidelines, authentication/authorization models, and data sensitivity.
    Provide the project full path, the component ID, the scan ID, and at least one security
    guideline.

    Example:
        ascp_create_security_context(
            project_path="my-group/my-project",
            component_id="gid://gitlab/Ascp::Component/1",
            scan_id="gid://gitlab/Ascp::Scan/1",
            guidelines=[{
                "name": "No direct DB access",
                "operation": "READ",
                "severity_if_violated": "HIGH"
            }],
            summary="Security context for the authentication service"
        )
    """
    args_schema: Type[BaseModel] = CreateAscpSecurityContextInput

    def format_display_message(
        self, args: CreateAscpSecurityContextInput, _tool_response: Any = None
    ) -> str:
        return f"Create ASCP security context for component {args.component_id}"

    async def _execute(self, **kwargs: Any) -> str:
        input_data = CreateAscpSecurityContextInput.model_validate(kwargs).model_dump(
            by_alias=True, exclude_none=True
        )
        variables = {"input": input_data}

        response = await self.gitlab_client.graphql(
            CREATE_ASCP_SECURITY_CONTEXT_MUTATION,
            variables,
        )

        if not isinstance(response, dict):
            raise ToolException("GraphQL returned no response or invalid format")

        graphql_errors = response.get("errors")
        if graphql_errors:
            messages = parse_graphql_errors(graphql_errors)
            exc_message = "; ".join(messages)
            raise ToolException(exc_message)

        payload = response.get("ascpSecurityContextCreate") or {}

        security_context = payload.get("securityContext")
        errors = payload.get("errors")

        if errors:
            if not isinstance(errors, list):
                errors = [str(errors)]
            raise ToolException("; ".join(errors))

        if not security_context or not security_context.get("id"):
            raise ToolException("Failed to create ASCP security context.")

        return CreateAscpSecurityContextResponse(
            security_context=security_context
        ).model_dump_json()
