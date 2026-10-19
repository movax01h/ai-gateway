"""Errors raised when a request asks for a model it may not use.

None of these subclass ``ValueError``: ``ModelConfigMiddleware``, the Duo Workflow Service
``ModelMetadataInterceptor`` and several API routes catch ``ValueError`` and fall back to a default
model. A denial must reach the client instead of being replaced silently.
"""

__all__ = [
    "FeatureSettingRequiredError",
    "ModelNotAllowedForFeatureError",
    "ModelSelectionError",
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
