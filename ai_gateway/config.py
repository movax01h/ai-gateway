import os
from typing import Annotated, Literal, Optional, Set, Tuple, TypedDict, Union
from urllib.parse import urlparse

import httpx
import litellm
from dotenv import find_dotenv
from pydantic import BaseModel, Field, Json, RootModel, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "Config",
    "ConfigAuditEvent",
    "ConfigAuth",
    "ConfigBedrockGuardrail",
    "ConfigCachingProxy",
    "ConfigCustomModels",
    "ConfigCustomersDot",
    "ConfigDuoChat",
    "ConfigDuoWorkflow",
    "ConfigFastApi",
    "ConfigGoogleCloudProfiler",
    "ConfigInstrumentator",
    "ConfigLogging",
    "ConfigMockUsageQuotaServer",
    "ConfigModelKeys",
    "ConfigModelLimits",
    "ConfigModelSelection",
    "ConfigMtls",
    "ConfigProcessLevelFeatureFlags",
    "ConfigProxyEndpoints",
    "ConfigSnowplow",
    "ConfigTLS",
    "ConfigVertexTextModel",
    "get_config",
]

ENV_PREFIX = "AIGW"


class ConfigLogging(BaseModel):
    level: str = "INFO"
    format_json: bool = True
    to_file: Optional[str] = None
    enable_request_logging: bool = False
    enable_litellm_logging: bool = False


class ConfigSelfSignedJwt(BaseModel):
    signing_key: str = ""
    validation_key: str = ""


class ConfigTLS(BaseModel):
    """TLS configuration for enabling HTTPS / gRPCS."""

    enabled: bool = False
    cert_file: Optional[str] = None
    key_file: Optional[str] = None

    @model_validator(mode="after")
    def validate_when_enabled(self) -> "ConfigTLS":
        if not self.enabled:
            return self
        if not (self.cert_file and self.key_file):
            raise ValueError(
                "cert_file and key_file must both be set when TLS is enabled"
            )
        if not os.path.isfile(self.cert_file):
            raise ValueError(f"cert_file does not point to a file: {self.cert_file}")
        if not os.path.isfile(self.key_file):
            raise ValueError(f"key_file does not point to a file: {self.key_file}")
        return self


class ConfigMtls(BaseModel):
    """Client-certificate (mutual TLS) auth for upstream model endpoints.

    Applies to any endpoint LiteLLM reaches through the OpenAI/Azure SDK client
    (self-hosted OpenAI-compatible proxies, Azure OpenAI, custom models). LiteLLM
    exposes no per-request hook for a client certificate on that path, so the cert
    is installed process-wide via litellm.client_session / litellm.aclient_session
    in setup_litellm.

    `verify` / `ca_bundle` control server-certificate trust and are independent of
    the client certificate: httpx does not read SSL_CERT_FILE / REQUESTS_CA_BUNDLE,
    so a corporate CA must be supplied explicitly via `ca_bundle`.
    """

    enabled: bool = Field(
        default=False,
        description="Enable client-certificate (mTLS) auth for upstream model endpoints.",
    )
    cert_file: Optional[str] = Field(
        default=None,
        description="Path to the client certificate PEM (cert only, or combined cert+key).",
    )
    key_file: Optional[str] = Field(
        default=None,
        description="Path to the client private key, if not bundled into cert_file.",
    )
    key_password: Optional[SecretStr] = Field(
        default=None,
        description="Password for an encrypted client private key.",
    )
    verify: bool = Field(
        default=True,
        description="Verify the upstream server certificate.",
    )
    ca_bundle: Optional[str] = Field(
        default=None,
        description="CA bundle to verify the upstream server certificate (e.g. a corporate CA).",
    )

    @model_validator(mode="after")
    def validate_when_enabled(self) -> "ConfigMtls":
        if not self.enabled:
            return self
        if not self.cert_file:
            raise ValueError("cert_file must be set when mTLS is enabled")
        if not os.path.isfile(self.cert_file):
            raise ValueError(f"cert_file does not point to a file: {self.cert_file}")
        if self.key_file and not os.path.isfile(self.key_file):
            raise ValueError(f"key_file does not point to a file: {self.key_file}")
        if self.key_password and not self.key_file:
            raise ValueError("key_file must be set when key_password is provided")
        if self.ca_bundle and not os.path.isfile(self.ca_bundle):
            raise ValueError(f"ca_bundle does not point to a file: {self.ca_bundle}")
        return self


class ConfigFastApi(BaseModel):
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    metrics_host: str = "0.0.0.0"
    metrics_port: int = 8082
    uvicorn_logger: dict = {"version": 1, "disable_existing_loggers": False}
    docs_url: Optional[str] = None
    openapi_url: Optional[str] = None
    redoc_url: Optional[str] = None
    reload: bool = False
    tls: ConfigTLS = Field(default_factory=ConfigTLS)


class ConfigAuth(BaseModel):
    bypass_external: bool = False
    bypass_external_with_header: bool = False
    bypass_jwt_signature: bool = False


class ConfigProcessLevelFeatureFlags(BaseModel):
    duo_classic_chat_duo_core_cutoff: bool = False


class ConfigGoogleCloudProfiler(BaseModel):
    enabled: bool = False
    verbose: int = 2
    period_ms: int = 10


class ConfigInstrumentator(BaseModel):
    thread_monitoring_enabled: bool = False
    thread_monitoring_interval: int = 60


class ConfigInternalEvent(BaseModel):
    enabled: bool = False
    app_id: str = "gitlab_ai_gateway"
    namespace: str = "gl"
    endpoint: Optional[str] = None
    batch_size: Optional[int] = 1
    thread_count: Optional[int] = 1


class ConfigBillingEvent(BaseModel):
    enabled: bool = False
    app_id: str = "gitlab_ai_gateway-billing"
    namespace: str = "gl"
    endpoint: Optional[str] = None
    batch_size: Optional[int] = 1
    thread_count: Optional[int] = 1


class ConfigAuditEvent(BaseModel):
    enabled: bool = False
    buffer_size: int = 100
    flush_interval_seconds: float = 10.0
    max_retries: int = 3


# TODO: Migrate to InternalEvent
# See https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/698
class ConfigSnowplow(ConfigInternalEvent):
    enabled: bool = False
    endpoint: Optional[str] = None
    batch_size: Optional[int] = 1
    thread_count: Optional[int] = 1


class ConfigCustomModels(BaseModel):
    enabled: bool = False
    disable_streaming: bool = False
    extra_headers: dict[str, str] | None = None


class ConfigDuoChat(BaseModel):
    max_tokens: Optional[int] = None


class ConfigAbuseDetection(BaseModel):
    enabled: bool = False
    sampling_rate: float = 0.1  # 1/10 of requests are sampled


class ConfigCustomersDot(BaseModel):
    api_user: Optional[str] = None
    api_token: Optional[str] = None


class ConfigAgenticMock(BaseModel):
    auto_tool_approval: Optional[bool] = False
    use_last_human_message: Optional[bool] = True


class ConfigMockUsageQuotaServer(BaseModel):
    """Configuration for the mock CustomersDot usage quota server.

    When ``AIGW_MOCK_USAGE_CREDITS=true`` the server is started automatically alongside AIGW and DWS
    """

    url: str = Field(
        default="http://localhost:4567",
        description=(
            "URL where the mock CustomersDot usage quota server listens. "
            "Used only when AIGW_MOCK_USAGE_CREDITS=true."
        ),
    )

    @property
    def port(self) -> int:
        """Port parsed from the configured ``url``.

        Falls back to the URL scheme's default port (80 for http, 443 for https) when the URL has no explicit port.
        """
        parsed = urlparse(self.url)
        if parsed.port is not None:
            return parsed.port
        return 443 if parsed.scheme == "https" else 80


class ConfigModelKeys(BaseModel):
    mistral_api_key: Optional[str] = None
    fireworks_provider_api_key: Optional[str] = None


def _build_location(default: str = "us-central1") -> str:
    """Reads the GCP region from the environment.

    Returns the default argument when not configured.
    """
    # pylint: disable=direct-environment-variable-reference
    return os.getenv("RUNWAY_REGION", default)
    # pylint: enable=direct-environment-variable-reference


def _build_endpoint() -> str:
    """Returns the default endpoint for Vertex AI.

    This code assumes that the Runway region (i.e. Cloud Run region) is the same as the Vertex AI region. To support
    other Cloud Run regions, this code will need to be updated to map to a nearby Vertex AI region instead.
    """
    return f"{_build_location()}-aiplatform.googleapis.com"


class ConfigGoogleCloudPlatform(BaseModel):
    project: str = ""
    service_account_json_key: str = ""
    location: str = Field(default_factory=_build_location)


class ConfigVertexTextModel(ConfigGoogleCloudPlatform):
    endpoint: str = Field(default_factory=_build_endpoint)


class ConfigVertexSearch(ConfigGoogleCloudPlatform):
    fallback_datastore_version: str = ""


class ConfigAmazonQ(BaseModel):
    region: str = ""
    endpoint_url: str = ""


class ConfigBindToolsCache(BaseModel):
    enabled: bool = False
    max_size: int = 128


class ConfigBedrockGuardrail(BaseModel):
    """Bedrock guardrail configuration.

    See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_GuardrailConfiguration.html
    """

    guardrailIdentifier: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2048,
            pattern=r"^([a-z0-9]+|arn:aws(-[^:]+)?:bedrock:[a-z0-9-]{1,20}:[0-9]{12}:guardrail/[a-z0-9]+)$",
            description="The guardrail identifier (short ID or full ARN).",
        ),
    ]
    guardrailVersion: Optional[
        Annotated[
            str,
            Field(
                pattern=r"^(|([1-9][0-9]{0,7})|(DRAFT))$",
                description="The guardrail version ('DRAFT' or a numeric string).",
            ),
        ]
    ] = None
    trace: Literal["enabled", "disabled"] = Field(
        default="disabled",
        description="Whether to enable guardrail trace in Bedrock responses.",
    )


class ConfigModelSelection(BaseModel):
    default_models: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "JSON object mapping feature_setting names to a list of model identifiers. "
            "Overrides the default_models from unit_primitives.yml for matching keys. "
            'Example: \'{"duo_chat": ["claude_sonnet_4_5_20250929_vertex"]}\''
        ),
    )
    model_params: dict[str, dict] = Field(
        default_factory=dict,
        description=(
            "JSON object whose keys are gitlab_identifier values from models.yml, "
            "and whose values are param dicts to be merged into the corresponding model's params. "
            'Example: \'{"claude_sonnet_4_20250514_bedrock": '
            '{"model_id": "arn:aws:bedrock:us-east-2:681816819199:application-inference-profile/oitsuvtb0pij"}}\''
        ),
    )
    prompt_params: dict[str, dict] = Field(
        default_factory=dict,
        description=(
            "JSON object whose keys are gitlab_identifier values from models.yml, "
            "and whose values are prompt_param dicts to be merged into the corresponding "
            "model's prompt_params. Use this for params that must be passed at model "
            "invocation rather than initialization. "
            'Example: \'{"claude_sonnet_4_5_20250929_vertex": {"vertex_location": "us-east5"}}\''
        ),
    )


class ConfigFeatureFlags(BaseModel):
    disallowed_flags: dict[str, Set[str]] = {}
    excl_post_process: list[str] = []
    fireworks_score_threshold: dict[str, float] = {}


class ConfigProxyEndpoints(BaseModel):
    """Configuration for allowing proxy endpoints per realm (allowlist model).

    Each realm is controlled by a single environment variable accepting a
    comma-separated list of IDs:

    - Absent or empty string → all instances/namespaces are allowed (no restriction).
    - ``"id1,id2"`` → only those instance UIDs (self-managed) or root namespace IDs (SaaS) are allowed.

    Environment variables::

        # Allow only specific self-managed instance UIDs
        AIGW_PROXY_ENDPOINTS__SELF_MANAGED_ENABLED="uid-a,uid-b"

        # Allow only specific SaaS root namespace IDs
        AIGW_PROXY_ENDPOINTS__SAAS_ENABLED="12345,67890"
    """

    self_managed_enabled: str = Field(
        default="",
        description=(
            "Controls which self-managed instances have proxy endpoints enabled. "
            "Comma-separated instance UIDs allow only those instances; "
            "empty string (default) allows all."
        ),
    )
    saas_enabled: str = Field(
        default="",
        description=(
            "Controls which SaaS namespaces have proxy endpoints enabled. "
            "Comma-separated root namespace IDs allow only those namespaces; "
            "empty string (default) allows all."
        ),
    )

    def self_managed_enabled_ids(self) -> list[str]:
        """Return the parsed allowlist of self-managed instance UIDs.

        Returns an empty list when the value is absent or empty (meaning all allowed).
        """
        return [v.strip() for v in self.self_managed_enabled.split(",") if v.strip()]

    def saas_enabled_ids(self) -> list[str]:
        """Return the parsed allowlist of SaaS root namespace IDs.

        Returns an empty list when the value is absent or empty (meaning all allowed).
        """
        return [v.strip() for v in self.saas_enabled.split(",") if v.strip()]


class ModelLimits(TypedDict, total=False):
    concurrency: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ConfigModelLimits(RootModel):
    root: dict[str, dict[str, ModelLimits]] = {}

    def for_model(self, engine: str, name: str) -> Optional[ModelLimits]:
        return self.root.get(engine, {}).get(name, None)


class ConfigCachingProxy(BaseSettings):
    """Configuration for the caching proxy used during load testing."""

    model_config = SettingsConfigDict(
        env_prefix="DUO_WORKFLOW_CACHING_PROXY_",
    )

    url: str = Field(
        default="http://localhost:8888",
        description="URL of the caching proxy server",
    )


class ConfigDuoWorkflow(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DUO_WORKFLOW_",
        env_nested_delimiter="__",
    )

    use_caching_proxy: bool = Field(
        default=False,
        description="Enable routing requests through caching proxy for load testing",
    )
    caching_proxy: ConfigCachingProxy = Field(default_factory=ConfigCachingProxy)
    tls: ConfigTLS = Field(default_factory=ConfigTLS)

    def caching_proxy_url(self) -> str | None:
        if self.use_caching_proxy:
            return self.caching_proxy.url
        return None


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_prefix=f"{ENV_PREFIX}_",
        protected_namespaces=(),
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "production"
    gitlab_url: str = "https://gitlab.com"
    gitlab_api_url: str = "https://gitlab.com/api/v4/"
    customer_portal_url: str = "https://customers.gitlab.com"
    glgo_base_url: str = "http://auth.token.gitlab.com"
    fireworks_api_base_url: str = "https://api.fireworks.ai/inference/v1"
    cloud_connector_service_name: str = "gitlab-ai-gateway"
    mock_model_responses: bool = False
    mock_usage_credits: bool = Field(
        default=False,
        description=(
            "When true, starts an in-process mock CustomersDot usage quota server "
            "so local development does not require a real CustomersDot. "
            "Intended for development only — never enable in production."
        ),
    )
    use_agentic_mock: bool = False
    bedrock_guardrail_config: Optional[Json[ConfigBedrockGuardrail]] = Field(
        default=None,
        description=(
            "Bedrock guardrail configuration as a JSON string. "
            "Set via AIGW_BEDROCK_GUARDRAIL_CONFIG_JSON. "
            'Example: \'{"guardrailIdentifier": "abc123", "guardrailVersion": "1", "trace": "disabled"}\''
        ),
    )

    logging: Annotated[ConfigLogging, Field(default_factory=ConfigLogging)]
    self_signed_jwt: Annotated[
        ConfigSelfSignedJwt, Field(default_factory=ConfigSelfSignedJwt)
    ]
    fastapi: Annotated[ConfigFastApi, Field(default_factory=ConfigFastApi)]
    auth: Annotated[ConfigAuth, Field(default_factory=ConfigAuth)]
    process_level_feature_flags: Annotated[
        ConfigProcessLevelFeatureFlags,
        Field(default_factory=ConfigProcessLevelFeatureFlags),
    ]
    google_cloud_profiler: Annotated[
        ConfigGoogleCloudProfiler, Field(default_factory=ConfigGoogleCloudProfiler)
    ]
    instrumentator: Annotated[
        ConfigInstrumentator, Field(default_factory=ConfigInstrumentator)
    ]
    snowplow: Annotated[ConfigSnowplow, Field(default_factory=ConfigSnowplow)]
    internal_event: Annotated[
        ConfigInternalEvent, Field(default_factory=ConfigInternalEvent)
    ]
    billing_event: Annotated[
        ConfigBillingEvent, Field(default_factory=ConfigBillingEvent)
    ]
    audit_event: Annotated[ConfigAuditEvent, Field(default_factory=ConfigAuditEvent)]
    google_cloud_platform: Annotated[
        ConfigGoogleCloudPlatform, Field(default_factory=ConfigGoogleCloudPlatform)
    ]
    amazon_q: Annotated[ConfigAmazonQ, Field(default_factory=ConfigAmazonQ)]
    custom_models: Annotated[
        ConfigCustomModels, Field(default_factory=ConfigCustomModels)
    ]
    mtls: Annotated[ConfigMtls, Field(default_factory=ConfigMtls)]
    duo_chat: Annotated[ConfigDuoChat, Field(default_factory=ConfigDuoChat)]
    model_keys: Annotated[ConfigModelKeys, Field(default_factory=ConfigModelKeys)]
    vertex_text_model: Annotated[
        ConfigVertexTextModel, Field(default_factory=ConfigVertexTextModel)
    ]
    vertex_search: Annotated[
        ConfigVertexSearch, Field(default_factory=ConfigVertexSearch)
    ]
    model_engine_limits: Annotated[
        ConfigModelLimits, Field(default_factory=ConfigModelLimits)
    ]
    model_selection: Annotated[
        ConfigModelSelection, Field(default_factory=ConfigModelSelection)
    ]
    bind_tools_cache: Annotated[
        ConfigBindToolsCache, Field(default_factory=ConfigBindToolsCache)
    ]
    feature_flags: Annotated[
        ConfigFeatureFlags, Field(default_factory=ConfigFeatureFlags)
    ]
    customersdot: Annotated[
        ConfigCustomersDot, Field(default_factory=ConfigCustomersDot)
    ]
    agentic_mock: Annotated[ConfigAgenticMock, Field(default_factory=ConfigAgenticMock)]
    duo_workflow: Annotated[ConfigDuoWorkflow, Field(default_factory=ConfigDuoWorkflow)]
    mock_usage_quota_server: Annotated[
        ConfigMockUsageQuotaServer, Field(default_factory=ConfigMockUsageQuotaServer)
    ]
    proxy_endpoints: Annotated[
        ConfigProxyEndpoints, Field(default_factory=ConfigProxyEndpoints)
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._apply_global_configs(
            parent=self.google_cloud_platform,
            children=[self.vertex_text_model, self.vertex_search],
        )

        # pylint: disable=direct-environment-variable-reference
        os.environ["CLOUD_CONNECTOR_SERVICE_NAME"] = self.cloud_connector_service_name
        # pylint: enable=direct-environment-variable-reference

    def _apply_global_configs(self, parent: BaseModel, children: list[BaseModel]):
        """Set a parent config to child configs if the field value is not specified."""
        for field in parent.model_fields_set:
            parent_value = getattr(parent, field)

            if not parent_value:
                continue

            for child in children:
                if field in child.model_fields_set:
                    continue

                setattr(child, field, parent_value)


_config = Config()


def get_config() -> Config:
    return _config


def setup_litellm(config: Config):
    litellm.vertex_project = config.google_cloud_platform.project
    # GitLab's AIGW deployment uses "global", but self-hosted Duo customers may use
    # a different location for vertex. Since the initialization path for vertex via litellm is
    # this method, we need to consider both cases.
    # Presence of VERTEXAI_LOCATION will take precedence (and hence work for self-hosted customers)
    # while "global" works for GitLab's AIGW deployments.
    # pylint: disable=direct-environment-variable-reference
    litellm.vertex_location = os.getenv("VERTEXAI_LOCATION") or "global"
    # pylint: enable=direct-environment-variable-reference

    _setup_litellm_mtls(config.mtls)


def _setup_litellm_mtls(mtls: ConfigMtls) -> None:
    """Install a client certificate for mutual TLS with upstream model endpoints.

    litellm.client_session / litellm.aclient_session is the only hook LiteLLM
    consults when building the OpenAI/Azure SDK client, which is the path used for
    self-hosted OpenAI-compatible and Azure endpoints. Setting them process-wide is
    harmless for other endpoints: a client certificate is only sent when the server
    requests one during the TLS handshake.
    """
    if not mtls.enabled:
        return

    # cert_file presence is guaranteed by ConfigMtls.validate_when_enabled.
    assert mtls.cert_file is not None

    # httpx `cert`: single PEM, (cert, key), or (cert, key, password).
    # key_password without key_file is rejected by ConfigMtls.validate_when_enabled.
    cert: Union[str, Tuple[str, str], Tuple[str, str, str]]
    if mtls.key_file and mtls.key_password:
        cert = (mtls.cert_file, mtls.key_file, mtls.key_password.get_secret_value())
    elif mtls.key_file:
        cert = (mtls.cert_file, mtls.key_file)
    else:
        cert = mtls.cert_file

    # httpx `verify` (server trust): explicit CA bundle > disabled > certifi default.
    verify: Union[bool, str]
    if not mtls.verify:
        verify = False
    elif mtls.ca_bundle:
        verify = mtls.ca_bundle
    else:
        verify = True

    # Installed once at startup and reused for the process lifetime; cleaned up on
    # process exit. setup_litellm is not re-invoked at runtime, so any previous
    # sessions are intentionally not closed here — they may still be serving
    # in-flight requests, and AsyncClient.aclose() cannot be awaited from here.
    litellm.client_session = httpx.Client(cert=cert, verify=verify)
    litellm.aclient_session = httpx.AsyncClient(cert=cert, verify=verify)
