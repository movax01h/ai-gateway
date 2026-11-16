"""Errors raised when a request asks for a model it may not use.

None of these subclass ``ValueError``: ``ModelConfigMiddleware``, the Duo Workflow Service
``ModelMetadataInterceptor`` and several API routes catch ``ValueError`` and fall back to a default
model. A denial must reach the client instead of being replaced silently.
"""

from typing import Optional

__all__ = [
    "FeatureSettingRequiredError",
    "ModelNotAllowedForFeatureError",
    "ModelSelectionError",
    "PromptNotServedOverHttpError",
    "ProviderNotServedError",
]


class ModelSelectionError(Exception):
    """Base class for model selection denials (HTTP 403, gRPC ``PERMISSION_DENIED``)."""


class ModelNotAllowedForFeatureError(ModelSelectionError):
    """The requested model is not in the feature's allowlist."""

    def __init__(self, feature_setting: str, gitlab_identifier: str) -> None:
        super().__init__(
            f"Model '{gitlab_identifier}' is not available for '{feature_setting}'"
        )
        self.feature_setting = feature_setting
        self.gitlab_identifier = gitlab_identifier


class FeatureSettingRequiredError(ModelSelectionError):
    """A GitLab-managed model was named without the feature it is requested for."""

    def __init__(self, gitlab_identifier: str) -> None:
        super().__init__(
            f"feature_setting is required to request model '{gitlab_identifier}'"
        )
        self.gitlab_identifier = gitlab_identifier


class ProviderNotServedError(ModelSelectionError):
    """The request names a provider that a gateway without custom models does not serve."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"This gateway does not serve models from provider '{provider}'"
        )
        self.provider = provider


class PromptNotServedOverHttpError(ModelSelectionError):
    """A prompt route was called for a prompt version that declares no ``models_from``."""

    def __init__(self, prompt_id: str, prompt_version: Optional[str]) -> None:
        version = (
            f"version '{prompt_version}'" if prompt_version else "the default version"
        )
        super().__init__(
            f"Prompt '{prompt_id}', {version}, is not served over HTTP. The prompt routes serve only the prompt "
            "versions that declare `models_from`. This version runs only in the Duo Workflow Service or behind "
            "another route, so do not add `models_from` to call it. To call it over HTTP on a local AI Gateway, set "
            "AIGW_MODEL_SELECTION__FEATURE_ALLOWLIST_MODE_HTTP=log."
        )
        self.prompt_id = prompt_id
        self.prompt_version = prompt_version
