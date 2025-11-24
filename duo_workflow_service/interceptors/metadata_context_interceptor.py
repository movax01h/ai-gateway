import grpc
from gitlab_cloud_connector.auth import X_GITLAB_REALM_HEADER, X_GITLAB_VERSION_HEADER

from ai_gateway.instrumentators.model_requests import (
    client_type,
    gitlab_realm,
    gitlab_version,
)
from ai_gateway.instrumentators.model_requests import (
    language_server_version as language_server_version_context,
)
from lib.language_server import LanguageServerVersion
from lib.prompts.caching import (
    X_GITLAB_MODEL_PROMPT_CACHE_ENABLED,
    set_prompt_caching_enabled_to_current_request,
)
from lib.verbose_ai_logs import VERBOSE_AI_LOGS_HEADER, current_verbose_ai_logs_context


class MetadataContextInterceptor(grpc.aio.ServerInterceptor):
    """Generic interceptor that maps metadata headers to context variables.

    This interceptor consolidates the logic of reading values from request metadata
    and setting them into context variables or calling setter functions.

    Replaces the following simple interceptors:
    - ClientTypeInterceptor
    - GitLabRealmInterceptor
    - GitLabVersionInterceptor
    - LanguageServerVersionInterceptor
    - EnabledInstanceVerboseAiLogsInterceptor
    - PromptCachingInterceptor
    """

    X_GITLAB_CLIENT_TYPE_HEADER = "x-gitlab-client-type"
    X_GITLAB_LANGUAGE_SERVER_VERSION = "x-gitlab-language-server-version"

    async def intercept_service(
        self,
        continuation,
        handler_call_details,
    ):
        """Intercept incoming requests to propagate metadata to context."""
        metadata = dict(handler_call_details.invocation_metadata)

        # Client type
        if value := metadata.get(self.X_GITLAB_CLIENT_TYPE_HEADER):
            client_type.set(value)

        # GitLab realm
        if value := metadata.get(X_GITLAB_REALM_HEADER.lower()):
            gitlab_realm.set(value)

        # GitLab version
        if value := metadata.get(X_GITLAB_VERSION_HEADER.lower()):
            gitlab_version.set(value)

        # Language server version
        if value := metadata.get(self.X_GITLAB_LANGUAGE_SERVER_VERSION):
            language_server_version_context.set(
                LanguageServerVersion.from_string(value)
            )

        # Verbose AI logs (always set, defaults to False)
        is_enabled = metadata.get(VERBOSE_AI_LOGS_HEADER) == "true"
        current_verbose_ai_logs_context.set(is_enabled)

        # Prompt caching (always called)
        set_prompt_caching_enabled_to_current_request(
            metadata.get(X_GITLAB_MODEL_PROMPT_CACHE_ENABLED.lower())
        )

        return await continuation(handler_call_details)
