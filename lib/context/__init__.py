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
from lib.context.model import current_model_metadata_context
from lib.context.request_metadata import (
    METADATA_LABELS,
    LLMFinishReason,
    build_metadata_labels,
    client_capabilities,
    client_type,
    gitlab_realm,
    gitlab_version,
    language_server_version,
)

__all__ = [
    # request_metadata
    "client_capabilities",
    "client_type",
    "gitlab_realm",
    "gitlab_version",
    "language_server_version",
    "METADATA_LABELS",
    "build_metadata_labels",
    "LLMFinishReason",
    # llm_operations
    "token_usage",
    "llm_operations",
    "TokenUsage",
    "LlmOperations",
    "init_token_usage",
    "get_token_usage",
    "init_llm_operations",
    "get_llm_operations",
    # auth
    "cloud_connector_token_context_var",
    "StarletteUser",
    "get_current_user",
    # model
    "current_model_metadata_context",
]
