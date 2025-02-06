from fastapi import Request

from ai_gateway.api.middleware import X_GITLAB_LANGUAGE_SERVER_VERSION
from ai_gateway.api.v2.code.typing import CompletionsRequestWithVersion
from ai_gateway.code_suggestions.language_server import LanguageServerVersion


class BaseModelProviderHandler:
    def __init__(
        self,
        payload: CompletionsRequestWithVersion,
        request: Request,
        completion_params: dict,
    ):
        self.payload = payload
        self.request = request
        self.completion_params = completion_params

    def update_completion_params(self):
        """Updates the completion_params dictionary in place with specific configurations."""

    def _update_code_context(self):
        self.completion_params.update(
            {"code_context": [ctx.content for ctx in self.payload.context]}
        )


class AnthropicHandler(BaseModelProviderHandler):
    def update_completion_params(self):
        # We support the prompt version 3 only with the Anthropic models
        if self.payload.prompt_version == 3:
            self.completion_params.update({"raw_prompt": self.payload.prompt})


class LiteLlmHandler(BaseModelProviderHandler):
    def update_completion_params(self):
        if self.payload.context:
            self._update_code_context()


class FireworksHandler(BaseModelProviderHandler):
    def update_completion_params(self):
        self.completion_params.update(
            {"max_output_tokens": 48, "context_max_percent": 0.3}
        )

        if self.payload.context:
            self._update_code_context()


class LegacyHandler(BaseModelProviderHandler):
    def update_completion_params(self):
        if self.payload.choices_count > 0:
            self.completion_params.update(
                {"candidate_count": self.payload.choices_count}
            )

        language_server_version = LanguageServerVersion.from_string(
            self.request.headers.get(X_GITLAB_LANGUAGE_SERVER_VERSION, None)
        )
        if language_server_version.supports_advanced_context() and self.payload.context:
            self._update_code_context()
