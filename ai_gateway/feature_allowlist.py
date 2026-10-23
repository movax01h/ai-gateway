"""Check client-supplied model metadata against the requested feature's allowlist.

GitLab Rails offers only the models that ``/v1/models/definitions`` lists for a feature. This
module applies the same lists on the server, where client-supplied model metadata enters: the
``model_metadata`` field of an HTTP body, the ``x-gitlab-agent-platform-model-metadata`` gRPC
header, and the routes that build metadata from their own payload, such as the legacy
``model_provider`` and ``model_name`` fields of ``/v2/code/completions``.

The allowlist is the one of the feature that the call site serves. A call site that passes no feature
uses the client's ``feature_setting``, so the client can name the feature whose allowlist suits it.
The HTTP prompt routes pass the feature of a prompt with an entry in ``PROMPT_ID_TO_FEATURE_SETTING``.
The fallback stays for any other prompt, and on the Duo Workflow Service until it derives the feature
from the flow. For a prompt without an entry, ``note_prompt_feature`` logs where the feature derived
from the prompt ID would not allow the requested model.

A gateway without custom models serves GitLab-managed models only. Over HTTP it accepts ``gitlab``
and ``amazon_q``, which GitLab Rails sends, and ``anthropic``, ``vertex-ai`` and ``fireworks_ai``,
which the legacy code completions fields send. Over gRPC it accepts ``gitlab`` and ``amazon_q``, and
the server-built ``fireworks_ai`` and ``mistral`` shapes that the Duo Workflow Service resume replays.
Any other provider is rejected: a custom provider string resolves a managed definition all the same,
and would otherwise skip the allowlist. In ``log`` mode the remaining checks still run for such a
request, so the log shows every mismatch. Amazon Q carries its own role and is not checked. Metadata
that the server builds itself is never checked. A gateway that serves custom models
(``AIGW_CUSTOM_MODELS__ENABLED``) runs every model on the operator's own keys, so it skips the check
altogether.

``AIGW_MODEL_SELECTION__FEATURE_ALLOWLIST_MODE`` selects the behavior:

- ``off``: no check.
- ``log``: log mismatches, change nothing (default).
- ``fallback``: serve the feature's default model instead of a model that is not allowed.
- ``enforce``: reject a model that is not allowed (HTTP 403, gRPC ``PERMISSION_DENIED``).

``fallback`` and ``enforce`` also rebuild a GitLab-managed model from its definition when a raw
``identifier`` would rewrite its wire model. ``off`` and ``log`` leave that override in place until
GitLab-managed models are resolved by their GitLab identifier only.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional

import structlog
from pydantic import BaseModel

from ai_gateway.config import FeatureAllowlistMode, get_config
from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
    resolve_provider_aware_metadata,
)
from ai_gateway.model_selection import (
    FeatureSettingRequiredError,
    LLMDefinition,
    ModelNotAllowedForFeatureError,
    ModelSelectionConfig,
    ModelSelectionError,
    ProviderNotServedError,
    UnitPrimitiveConfig,
)
from ai_gateway.models.base import SERVER_KEYED_PROVIDERS
from lib.context import StarletteUser

__all__ = [
    "SERVED_PROVIDERS",
    "ClientModelRequest",
    "check_client_model_metadata",
    "note_prompt_feature",
]

log = structlog.stdlib.get_logger("feature_allowlist")

Surface = Literal["http", "dws"]

# Providers a gateway without custom models serves, per surface. GitLab Rails sends `gitlab`
# and `amazon_q`. The legacy `model_provider` field of `/v2/code/completions` sends `anthropic`,
# `vertex-ai` and `fireworks_ai`. The Duo Workflow Service resume replays the server-built
# `fireworks_ai` and `mistral` shapes of a checkpointed model. Self-hosted providers (`openai`,
# `litellm`, ...) only reach a gateway that serves custom models, which skips this check. Every
# other value, known or not, resolves a GitLab-managed definition through `create_model_metadata`
# and is rejected here.
SERVED_PROVIDERS: dict[Surface, frozenset[str]] = {
    "http": frozenset({"gitlab", "amazon_q", "anthropic", "vertex-ai", "fireworks_ai"}),
    "dws": frozenset({"gitlab", "amazon_q", "fireworks_ai", "mistral"}),
}

_LOG_VALUE_MAX_LENGTH = 255
_MISMATCH_EVENT = "Model allowlist mismatch"
_NOTE_EVENT = "Model allowlist note"


@dataclass(frozen=True)
class ClientModelRequest:
    """What the client asked for, read from the raw metadata before ``create_model_metadata`` consumes it.

    For a request without ``name``, ``create_model_metadata`` removes ``identifier`` and ``feature_setting`` from the
    mapping it gets and sets ``name``. Read the request first, and build the metadata from a copy of the raw value.
    """

    provider: Optional[str] = None
    identifier: Optional[str] = None
    name: Optional[str] = None
    feature_setting: Optional[str] = None

    @classmethod
    def from_raw(cls, raw: Any) -> "ClientModelRequest":
        """Build the request from the raw ``model_metadata`` value.

        A pydantic model is read through ``model_dump``. Any other value that is not a mapping
        yields an empty request. String values are cut to 255 characters, which is enough to
        identify a model and keeps log fields short.

        Args:
            raw: The ``model_metadata`` value as the client sent it.

        Returns:
            The parsed request.
        """
        if isinstance(raw, BaseModel):
            raw = raw.model_dump()
        if not isinstance(raw, Mapping):
            return cls()
        return cls(
            provider=_text(raw.get("provider")),
            identifier=_text(raw.get("identifier")),
            name=_text(raw.get("name")),
            feature_setting=_text(raw.get("feature_setting")),
        )

    @property
    def pinned(self) -> bool:
        """Whether the client named a model, by ``name`` or by ``identifier``."""
        return bool(self.name or self.identifier)


def _text(value: Any) -> Optional[str]:
    if isinstance(value, str) and value:
        return value[:_LOG_VALUE_MAX_LENGTH]
    return None


def _use_prompt_default() -> None:
    """Fallback for a request with nothing to fall back to: the prompt's own default model is used."""
    return None


def _outcome(
    mode: FeatureAllowlistMode,
    metadata: TypeModelMetadata,
    error: ModelSelectionError,
    fallback: Callable[[], Optional[TypeModelMetadata]],
) -> Optional[TypeModelMetadata]:
    if mode == "log":
        return metadata
    if mode == "fallback":
        return fallback()
    raise error


@dataclass(frozen=True)
class _AllowlistCheck:
    """One check of a managed model request, in a given mode."""

    mode: FeatureAllowlistMode
    surface: Surface
    request: ClientModelRequest
    feature: Optional[str]
    user: Optional[StarletteUser]
    using_cache: Optional[bool]
    log_context: Mapping[str, Any]
    fireworks_api_base_url: str
    mock_model_responses: bool
    # Server API keys. Keep them out of the repr, which an error report can capture.
    provider_keys: dict[str, Any] = field(repr=False)

    def apply(self, metadata: TypeModelMetadata) -> Optional[TypeModelMetadata]:
        if self.request.provider not in SERVED_PROVIDERS[self.surface]:
            self._provider_not_served(metadata)

        metadata = self._drop_identifier_override(metadata)
        if not self.request.pinned:
            return self._feature_default(metadata)

        if self.feature is None:
            return self._missing_feature(metadata)

        unit_primitive_config = self._unit_primitive_config(self.feature)
        if unit_primitive_config is None:
            return self._unknown_feature(metadata, self.feature)

        return self._check_allowlist(metadata, unit_primitive_config)

    def note_prompt_feature(self, metadata: TypeModelMetadata) -> None:
        """Log a note when a named model would not pass the allowlist of this check's feature.

        It changes nothing, in any mode. A request whose ``feature_setting`` is this feature
        already met the same allowlist in ``apply``, so it gets no note.
        """
        if (
            not self.request.pinned
            or self.feature is None
            or self.feature == self.request.feature_setting
        ):
            return

        unit_primitive_config = self._unit_primitive_config(self.feature)
        if unit_primitive_config is None:
            self._note("prompt_feature_unknown", metadata)
        elif (
            metadata.llm_definition.gitlab_identifier
            not in unit_primitive_config.allowed_model_identifiers
        ):
            self._note("prompt_feature_model_not_allowed", metadata)

    def _feature_default(self, metadata: TypeModelMetadata) -> TypeModelMetadata:
        """Serve the default of the feature the call site names, not of the one the client named.

        An unpinned request asks for "the feature default". ``create_model_metadata`` drew that
        default from the client's ``feature_setting``, which is the right one unless the call site
        passed a different feature. ``/v1/embeddings/code_embeddings`` with
        ``{"provider": "gitlab", "feature_setting": "duo_agent_platform"}`` would otherwise get the
        Agent Platform default as an embedding model. In ``fallback`` and ``enforce`` mode the
        default is drawn from the call-site feature instead; a mismatch is not refused, because
        legitimate clients send another feature's name on some paths today.
        """
        if self.feature is None or self.request.feature_setting == self.feature:
            return metadata
        self._warn("feature_mismatch", metadata)
        if self.mode == "log" or self._unit_primitive_config(self.feature) is None:
            return metadata
        return self._default_of(self.feature)

    def _provider_not_served(self, metadata: TypeModelMetadata) -> None:
        """Refuse a provider this gateway does not serve; there is nothing valid to fall back to.

        ``log`` mode records the provider and runs the remaining checks, so the log shows what the
        request would meet once the provider is fixed.
        """
        self._warn("provider_not_served", metadata)
        if self.mode != "log":
            raise ProviderNotServedError(self.request.provider or "")

    def _missing_feature(
        self, metadata: TypeModelMetadata
    ) -> Optional[TypeModelMetadata]:
        """Handle a named model without a feature, which the allowlist cannot be applied to."""
        if self.surface == "dws":
            # The Duo Workflow Service resume replays a checkpointed model as
            # `{"provider": "gitlab", "name": ...}`, without a feature. The flow-derived check
            # covers it once the flow is known.
            self._note("missing_feature_setting", metadata)
            return metadata
        self._warn("missing_feature_setting", metadata)
        return _outcome(
            self.mode,
            metadata,
            FeatureSettingRequiredError(metadata.llm_definition.gitlab_identifier),
            _use_prompt_default,
        )

    def _unknown_feature(
        self, metadata: TypeModelMetadata, feature: str
    ) -> Optional[TypeModelMetadata]:
        """Handle a named model whose feature does not exist, on both surfaces: no resume replays this shape."""
        self._warn("unknown_feature", metadata)
        return _outcome(
            self.mode,
            metadata,
            ModelNotAllowedForFeatureError(
                feature, metadata.llm_definition.gitlab_identifier
            ),
            _use_prompt_default,
        )

    @staticmethod
    def _unit_primitive_config(feature: str) -> Optional[UnitPrimitiveConfig]:
        configs = ModelSelectionConfig.instance()
        return configs.get_resolved_unit_primitive_config_map().get(feature)

    def _drop_identifier_override(
        self, metadata: TypeModelMetadata
    ) -> TypeModelMetadata:
        """Drop a raw ``identifier`` that would rewrite the wire model of a GitLab-managed request.

        The allowlist checks the model that ``name`` resolved to, but a request such as::

            {
                "provider": "gitlab",
                "name": "claude_sonnet_4_6_vertex",
                "identifier": "vertex_ai/claude-opus-4-8",
            }

        passes the allowlist as Sonnet and then runs Opus on the gateway's Vertex AI credentials, because
        ``to_params`` turns the identifier into ``custom_llm_provider="vertex_ai"`` and
        ``model="claude-opus-4-8"``. There is no endpoint or key for ``validate_custom_endpoint``
        to refuse. A client ``endpoint`` or ``api_key`` is not handled here: on a gateway without
        custom models ``validate_custom_endpoint`` already rejects both when the model is built.

        The override is read from the resolved metadata, so the resolution of each surface decides:
        ``create_model_metadata`` keeps a raw ``identifier`` only next to ``name``, and the Duo
        Workflow Service resolves a ``gitlab`` request by its ``identifier`` instead. The Mistral
        builder sets ``identifier`` itself, which is not an override.

        In ``fallback`` and ``enforce`` mode the metadata is rebuilt from the pristine model
        definition, so only the server decides the wire model and its limits:
        ``_create_gitlab_metadata`` copies limits and tuning from the models.yml entry the raw
        identifier matched into the resolved definition. ``log`` mode only records the attempt.
        """
        if (
            not isinstance(metadata, ModelMetadata)
            or not metadata.identifier
            or metadata.provider in SERVER_KEYED_PROVIDERS
        ):
            return metadata
        self._warn("managed_identifier_override", metadata)
        if self.mode == "log":
            return metadata
        return self._rebuild(
            ModelSelectionConfig.instance().get_model(
                metadata.llm_definition.gitlab_identifier
            )
        )

    def _check_allowlist(
        self, metadata: TypeModelMetadata, unit_primitive_config: UnitPrimitiveConfig
    ) -> Optional[TypeModelMetadata]:
        feature = unit_primitive_config.feature_setting
        gitlab_identifier = metadata.llm_definition.gitlab_identifier
        if gitlab_identifier not in unit_primitive_config.allowed_model_identifiers:
            self._warn("model_not_allowed", metadata)
            return _outcome(
                self.mode,
                metadata,
                ModelNotAllowedForFeatureError(feature, gitlab_identifier),
                lambda: self._default_of(feature),
            )

        if (
            unit_primitive_config.dev
            and gitlab_identifier in unit_primitive_config.dev.selectable_models
        ):
            self._note("dev_model", metadata)
        elif any(
            deprecated.identifier == gitlab_identifier
            for deprecated in unit_primitive_config.deprecated_models
        ):
            self._note("deprecated_model", metadata)
        return metadata

    def _default_of(self, feature: str) -> TypeModelMetadata:
        return self._rebuild(
            ModelSelectionConfig.instance().get_model_for_feature(feature)
        )

    def _rebuild(self, llm_definition: LLMDefinition) -> TypeModelMetadata:
        """Build server-side metadata for a model that the check chose.

        The metadata keeps the request settings that the server builders set: the user's global ID
        as the Fireworks session ID, the ``using_cache`` choice of the client metadata, and the
        mock-response mode.
        """
        return resolve_provider_aware_metadata(
            llm_definition,
            provider_keys=self.provider_keys,
            fireworks_api_base_url=self.fireworks_api_base_url,
            mock_model_responses=self.mock_model_responses,
            session_id=self.user.global_user_id if self.user else None,
            using_cache=self.using_cache,
        )

    def _fields(self, reason: str, metadata: TypeModelMetadata) -> dict[str, Any]:
        """Return the log fields; the event and the check's fields win over a caller key of the same name."""
        context = {
            key: value for key, value in self.log_context.items() if key != "event"
        }
        return {
            **context,
            "reason": reason,
            "mode": self.mode,
            "surface": self.surface,
            "feature_setting": self.feature,
            "client_feature_setting": self.request.feature_setting,
            "provider": self.request.provider,
            "requested_name": self.request.name,
            "requested_identifier": self.request.identifier,
            "pinned": self.request.pinned,
            "served_identifier": metadata.llm_definition.gitlab_identifier,
        }

    def _warn(self, reason: str, metadata: TypeModelMetadata) -> None:
        log.warning(_MISMATCH_EVENT, **self._fields(reason, metadata))

    def _note(self, reason: str, metadata: TypeModelMetadata) -> None:
        log.info(_NOTE_EVENT, **self._fields(reason, metadata))


def check_client_model_metadata(
    request: ClientModelRequest,
    metadata: TypeModelMetadata,
    *,
    surface: Surface,
    feature_setting: Optional[str] = None,
    user: Optional[StarletteUser] = None,
    log_context: Optional[Mapping[str, Any]] = None,
) -> Optional[TypeModelMetadata]:
    """Check client-supplied model metadata against the requested feature's allowlist.

    The check covers one metadata object. On the Duo Workflow Service the interceptor resolves a
    ``ModelMetadataByTag``, whose ``by_tag`` map and ``feature_setting`` also come from the client's
    ``feature_setting``: components select a model by tag from ``by_tag``, and the classifier
    routing rewrites the default from the tag models of ``feature_setting``. A caller that passes a
    feature other than the client's must rebuild both from that feature, not only ``default``.

    Args:
        request: What the client asked for, read with ``ClientModelRequest.from_raw`` before the
            metadata was built.
        metadata: The metadata that the surface resolved from the same raw value.
        surface: ``http`` for the ``model_metadata`` body field and the route payloads, ``dws``
            for the gRPC header.
        feature_setting: The feature the call site serves. It wins over the client's value. Without
            it the client's ``feature_setting`` selects the allowlist, so pass it whenever the call
            site knows its feature.
        user: The user the request runs for. Metadata that the check builds uses the user's global
            ID as the Fireworks session ID, as the server builders do.
        log_context: Extra fields for the log event, such as ``path``. The check's own fields win
            over a key with the same name.

    Returns:
        The metadata to use. ``None`` means "use the prompt's default model": ``fallback`` mode,
        for a named model without a feature on HTTP, or with an unknown feature.

    Raises:
        ModelSelectionError: In ``enforce`` mode, when the request is not allowed.
    """
    check = _new_check(
        request,
        metadata,
        surface=surface,
        feature=feature_setting or request.feature_setting,
        user=user,
        log_context=log_context,
    )
    return metadata if check is None else check.apply(metadata)


def note_prompt_feature(
    request: ClientModelRequest,
    metadata: TypeModelMetadata,
    *,
    prompt_feature_setting: str,
    log_context: Optional[Mapping[str, Any]] = None,
) -> None:
    """Log a note when a named model would not pass the allowlist of an HTTP prompt's own feature.

    The prompt routes call it for a prompt without an entry in ``PROMPT_ID_TO_FEATURE_SETTING``. For
    such a prompt, ``check_client_model_metadata`` still uses the client's ``feature_setting``,
    because the feature derived from a prompt ID does not always match the one that GitLab Rails
    sends. The note shows where the two differ, so the prompt can get an entry. It changes nothing,
    in any mode.

    Args:
        request: What the client asked for, read with ``ClientModelRequest.from_raw`` before the
            metadata was built.
        metadata: The metadata resolved from the same raw value, before the check.
        prompt_feature_setting: The feature derived from the prompt ID of the route.
        log_context: Extra fields for the log event, such as ``path`` and ``prompt_id``.
    """
    check = _new_check(
        request,
        metadata,
        surface="http",
        feature=prompt_feature_setting,
        log_context=log_context,
    )
    if check is not None:
        check.note_prompt_feature(metadata)


def _new_check(
    request: ClientModelRequest,
    metadata: TypeModelMetadata,
    *,
    surface: Surface,
    feature: Optional[str],
    user: Optional[StarletteUser] = None,
    log_context: Optional[Mapping[str, Any]] = None,
) -> Optional[_AllowlistCheck]:
    """Return the check for a request, or None when this gateway does not check it."""
    config = get_config()
    mode = config.model_selection.feature_allowlist_mode
    if (
        mode == "off"
        or config.custom_models.enabled
        or isinstance(metadata, AmazonQModelMetadata)
    ):
        return None

    return _AllowlistCheck(
        mode=mode,
        surface=surface,
        request=request,
        feature=feature,
        user=user,
        using_cache=getattr(metadata, "using_cache", None),
        log_context=log_context or {},
        fireworks_api_base_url=config.fireworks_api_base_url,
        mock_model_responses=config.mock_model_responses,
        provider_keys=config.model_keys.model_dump(),
    )
