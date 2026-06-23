# Context variables shared between ai_gateway and duo_workflow_service

from lib.context.auth import (
    StarletteUser,
    cloud_connector_token_context_var,
    get_current_user,
)
from lib.context.llm_operations import (
    LlmOperations,
    TokenUsage,
    get_llm_operations,
    get_token_usage,
    init_llm_operations,
    init_token_usage,
    llm_operations,
    token_usage,
)
from lib.context.model import (
    ModelSizeBucket,
    current_model_metadata_context,
    current_model_metadata_with_size_context,
    get_model_metadata,
)
from lib.context.orbit import (
    build_orbit_session_summary_extras,
    init_orbit_counters,
    is_orbit_tool,
    orbit_tool_call_count,
    total_tool_call_count,
)
from lib.context.request_metadata import (
    METADATA_LABELS,
    LLMFinishReason,
    build_metadata_labels,
    client_capabilities,
    client_type,
    extract_finish_reason,
    gitlab_instance_id,
    gitlab_realm,
    gitlab_version,
    is_gitlab_team_member,
    language_server_version,
)
from lib.context.tool_executions import (
    ToolExecutions,
    get_tool_executions,
    init_tool_executions,
    tool_executions,
)
from lib.context.workflow import (
    get_workflow_id,
    set_workflow_id,
)

__all__ = [
    # request_metadata
    "client_capabilities",
    "client_type",
    "gitlab_instance_id",
    "gitlab_realm",
    "gitlab_version",
    "is_gitlab_team_member",
    "language_server_version",
    "METADATA_LABELS",
    "build_metadata_labels",
    "LLMFinishReason",
    "extract_finish_reason",
    # llm_operations
    "token_usage",
    "llm_operations",
    "TokenUsage",
    "LlmOperations",
    "init_token_usage",
    "get_token_usage",
    "init_llm_operations",
    "get_llm_operations",
    # tool_executions
    "tool_executions",
    "init_tool_executions",
    "get_tool_executions",
    "ToolExecutions",
    # auth
    "cloud_connector_token_context_var",
    "StarletteUser",
    "get_current_user",
    # model
    "current_model_metadata_with_size_context",
    "current_model_metadata_context",
    "get_model_metadata",
    "ModelSizeBucket",
    # orbit
    "is_orbit_tool",
    "init_orbit_counters",
    "orbit_tool_call_count",
    "total_tool_call_count",
    "build_orbit_session_summary_extras",
    # workflow
    "get_workflow_id",
    "set_workflow_id",
]
