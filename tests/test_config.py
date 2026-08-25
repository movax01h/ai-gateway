# pylint: disable=direct-environment-variable-reference,too-many-lines
import os
import socket
import ssl
import threading
from unittest import mock

import litellm
import pytest
from pydantic import ValidationError

from ai_gateway.config import (
    Config,
    ConfigAmazonQ,
    ConfigAuditEvent,
    ConfigAuth,
    ConfigBedrockGuardrail,
    ConfigBillingEvent,
    ConfigCachingProxy,
    ConfigCustomModels,
    ConfigDuoChat,
    ConfigDuoWorkflow,
    ConfigFastApi,
    ConfigFeatureFlags,
    ConfigGoogleCloudPlatform,
    ConfigGoogleCloudProfiler,
    ConfigInstrumentator,
    ConfigLogging,
    ConfigMockUsageQuotaServer,
    ConfigModelLimits,
    ConfigModelSelection,
    ConfigMtls,
    ConfigProcessLevelFeatureFlags,
    ConfigSnowplow,
    ConfigTLS,
    ConfigVertexSearch,
    ConfigVertexTextModel,
    setup_litellm,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            {
                "AIGW_GITLAB_URL": "http://gitlab.test",
                "AIGW_GITLAB_API_URL": "http://api.gitlab.test",
                "AIGW_CUSTOMER_PORTAL_URL": "http://customer.gitlab.test",
                "AIGW_GLGO_BASE_URL": "http://auth.token.gitlab.com",
                "AIGW_MOCK_MODEL_RESPONSES": "true",
            },
            Config(
                gitlab_url="http://gitlab.test",
                gitlab_api_url="http://api.gitlab.test",
                customer_portal_url="http://customer.gitlab.test",
                glgo_base_url="http://auth.token.gitlab.com",
                mock_model_responses=True,
            ),
        ),
    ],
)
def test_config_base(values: dict, expected: Config):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None, _env_prefix="AIGW_")

        keys = {
            "gitlab_url",
            "gitlab_api_url",
            "customer_portal_url",
            "glgo_base_url",
            "mock_model_responses",
        }

        actual = config.model_dump(include=keys)
        assert actual == expected.model_dump(include=keys)
        assert len(actual) == len(keys)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigLogging()),
        (
            {
                "AIGW_LOGGING__LEVEL": "DEBUG",
                "AIGW_LOGGING__FORMAT_JSON": "no",
                "AIGW_LOGGING__TO_FILE": "/file/file1.text",
            },
            ConfigLogging(level="DEBUG", format_json=False, to_file="/file/file1.text"),
        ),
    ],
)
def test_config_logging(values: dict, expected: ConfigLogging):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.logging == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigFastApi()),
        (
            {
                "AIGW_FASTAPI__API_HOST": "localhost",
                "AIGW_FASTAPI__API_PORT": "80",
                "AIGW_FASTAPI__METRICS_HOST": "localhost",
                "AIGW_FASTAPI__METRICS_PORT": "82",
                "AIGW_FASTAPI__UVICORN_LOGGER": '{"key": "value"}',
                "AIGW_FASTAPI__DOCS_URL": "docs.test",
                "AIGW_FASTAPI__OPENAPI_URL": "openapi.test",
                "AIGW_FASTAPI__REDOC_URL": "redoc.test",
                "AIGW_FASTAPI__RELOAD": "True",
            },
            ConfigFastApi(
                api_host="localhost",
                api_port=80,
                metrics_host="localhost",
                metrics_port=82,
                uvicorn_logger={"key": "value"},
                docs_url="docs.test",
                openapi_url="openapi.test",
                redoc_url="redoc.test",
                reload=True,
            ),
        ),
    ],
)
def test_config_fastapi(values: dict, expected: ConfigFastApi):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.fastapi == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigAuth()),
        ({"AIGW_AUTH__BYPASS_EXTERNAL": "yes"}, ConfigAuth(bypass_external=True)),
    ],
)
def test_config_auth_bypass_external(values: dict, expected: ConfigAuth):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.auth == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigProcessLevelFeatureFlags()),
        (
            {
                "AIGW_PROCESS_LEVEL_FEATURE_FLAGS__DUO_CLASSIC_CHAT_DUO_CORE_CUTOFF": "yes"
            },
            ConfigProcessLevelFeatureFlags(duo_classic_chat_duo_core_cutoff=True),
        ),
    ],
)
def test_config_process_level_feature_flags(
    values: dict, expected: ConfigProcessLevelFeatureFlags
):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.process_level_feature_flags == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigAuth()),
        (
            {"AIGW_AUTH__BYPASS_EXTERNAL_WITH_HEADER": "yes"},
            ConfigAuth(bypass_external_with_header=True),
        ),
    ],
)
def test_config_auth_bypass_external_with_header(values: dict, expected: ConfigAuth):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.auth == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigGoogleCloudProfiler()),
        (
            {
                "AIGW_GOOGLE_CLOUD_PROFILER__ENABLED": "yes",
                "AIGW_GOOGLE_CLOUD_PROFILER__VERBOSE": "1",
                "AIGW_GOOGLE_CLOUD_PROFILER__PERIOD_MS": "5",
            },
            ConfigGoogleCloudProfiler(enabled=True, verbose=1, period_ms=5),
        ),
    ],
)
def test_config_google_cloud_profiler(
    values: dict, expected: ConfigGoogleCloudProfiler
):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.google_cloud_profiler == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigFeatureFlags()),
        (
            {
                "AIGW_FEATURE_FLAGS__EXCL_POST_PROCESS": '["func1", "func2"]',
                "AIGW_FEATURE_FLAGS__FIREWORKS_SCORE_THRESHOLD": '{"model": "-1.0"}',
            },
            ConfigFeatureFlags(
                excl_post_process=["func1", "func2"],
                fireworks_score_threshold={"model": -1.0},
            ),
        ),
    ],
)
def test_config_f_flags_code_suggestions(values: dict, expected: ConfigFeatureFlags):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.feature_flags == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigSnowplow()),
        (
            {
                "AIGW_SNOWPLOW__ENABLED": "yes",
                "AIGW_SNOWPLOW__ENDPOINT": "endpoint.test",
                "AIGW_SNOWPLOW__BATCH_SIZE": "8",
                "AIGW_SNOWPLOW__THREAD_COUNT": "7",
            },
            ConfigSnowplow(
                enabled=True, endpoint="endpoint.test", thread_count=7, batch_size=8
            ),
        ),
    ],
)
def test_config_snowplow(values: dict, expected: ConfigSnowplow):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.snowplow == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigInstrumentator()),
        (
            {
                "AIGW_INSTRUMENTATOR__THREAD_MONITORING_ENABLED": "True",
                "AIGW_INSTRUMENTATOR__THREAD_MONITORING_INTERVAL": "45",
            },
            ConfigInstrumentator(
                thread_monitoring_enabled=True, thread_monitoring_interval=45
            ),
        ),
    ],
)
def test_config_instrumentator(values: dict, expected: ConfigInstrumentator):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.instrumentator == expected


@pytest.mark.parametrize(
    (
        "values",
        "expected_google_cloud_platform",
        "expected_vertex_text_model",
        "expected_vertex_search",
    ),
    [
        (
            {},
            ConfigGoogleCloudPlatform(),
            ConfigVertexTextModel(),
            ConfigVertexSearch(),
        ),
        (
            {
                "AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT": "global-project",
                "AIGW_GOOGLE_CLOUD_PLATFORM__SERVICE_ACCOUNT_JSON_KEY": "global-secret",
            },
            ConfigGoogleCloudPlatform(
                project="global-project",
                service_account_json_key="global-secret",
            ),
            ConfigVertexTextModel(
                project="global-project",
                service_account_json_key="global-secret",
            ),
            ConfigVertexSearch(
                project="global-project",
                service_account_json_key="global-secret",
            ),
        ),
        (
            {
                "AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT": "global-project",
                "AIGW_VERTEX_TEXT_MODEL__PROJECT": "specific-project-1",
                "AIGW_VERTEX_SEARCH__PROJECT": "specific-project-2",
            },
            ConfigGoogleCloudPlatform(
                project="global-project",
            ),
            ConfigVertexTextModel(
                project="specific-project-1",
            ),
            ConfigVertexSearch(
                project="specific-project-2",
            ),
        ),
        (
            {
                "AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT": "global-project",
                "AIGW_VERTEX_TEXT_MODEL__PROJECT": "",
                "AIGW_VERTEX_SEARCH__PROJECT": "",
            },
            ConfigGoogleCloudPlatform(
                project="global-project",
            ),
            ConfigVertexTextModel(
                project="",
            ),
            ConfigVertexSearch(
                project="",
            ),
        ),
    ],
)
def test_config_google_cloud_platform(
    values: dict,
    expected_google_cloud_platform: ConfigGoogleCloudPlatform,
    expected_vertex_text_model: ConfigVertexTextModel,
    expected_vertex_search: ConfigVertexSearch,
):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.google_cloud_platform == expected_google_cloud_platform
        assert config.vertex_text_model == expected_vertex_text_model
        assert config.vertex_search == expected_vertex_search


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigCustomModels(enabled=False)),
        (
            {
                "AIGW_CUSTOM_MODELS__ENABLED": "True",
            },
            ConfigCustomModels(
                enabled=True,
            ),
        ),
    ],
)
def test_custom_models(values: dict, expected: ConfigCustomModels):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.custom_models == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigVertexTextModel()),
        (
            {
                "AIGW_VERTEX_TEXT_MODEL__PROJECT": "project",
                "AIGW_VERTEX_TEXT_MODEL__LOCATION": "location",
                "AIGW_VERTEX_TEXT_MODEL__ENDPOINT": "endpoint",
                "RUNWAY_REGION": "test-case1",  # ignored
            },
            ConfigVertexTextModel(
                project="project",
                location="location",
                endpoint="endpoint",
            ),
        ),
        (
            {
                "AIGW_VERTEX_TEXT_MODEL__PROJECT": "project",
                "RUNWAY_REGION": "test-case1",
            },
            ConfigVertexTextModel(
                project="project",
                location="test-case1",
                endpoint="test-case1-aiplatform.googleapis.com",
            ),
        ),
    ],
)
def test_config_vertex_text_model(values: dict, expected: ConfigVertexTextModel):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.vertex_text_model == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigVertexSearch()),
        (
            {
                "AIGW_VERTEX_SEARCH__PROJECT": "project",
            },
            ConfigVertexSearch(
                project="project",
                fallback_datastore_version="",
            ),
        ),
        (
            {
                "AIGW_VERTEX_SEARCH__PROJECT": "project",
                "AIGW_VERTEX_SEARCH__FALLBACK_DATASTORE_VERSION": "17.0",
            },
            ConfigVertexSearch(
                project="project",
                fallback_datastore_version="17.0",
            ),
        ),
    ],
)
def test_config_vertex_search(values: dict, expected: ConfigVertexSearch):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.vertex_search == expected


def test_amazon_q():
    values = {
        "AIGW_AMAZON_Q__REGION": "us-west-2",
        "AIGW_AMAZON_Q__ENDPOINT_URL": "https://us-west-2.gamma.integration.qdev.ai.aws.dev",
    }

    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.amazon_q == ConfigAmazonQ(
            region="us-west-2",
            endpoint_url="https://us-west-2.gamma.integration.qdev.ai.aws.dev",
        )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigModelLimits()),
        (
            {
                "AIGW_MODEL_ENGINE_LIMITS": '{"engine": {"model": {"input_tokens": 10, "output_tokens": 20, "concurrency": 30}}}'  # pylint: disable=line-too-long
            },
            ConfigModelLimits(
                {
                    "engine": {
                        "model": {
                            "input_tokens": 10,
                            "output_tokens": 20,
                            "concurrency": 30,
                        }
                    }
                }
            ),
        ),
        (
            {"AIGW_MODEL_ENGINE_LIMITS": '{"engine": {"model": {"concurrency": 30}}}'},
            ConfigModelLimits({"engine": {"model": {"concurrency": 30}}}),
        ),
        (
            {
                "AIGW_MODEL_ENGINE_LIMITS": '{"bedrock": {"anthropic.claude-3-sonnet-20240229-v1:0": {"total_tokens": 4096, "concurrency": 60}}}'  # pylint: disable=line-too-long
            },
            ConfigModelLimits(
                {
                    "bedrock": {
                        "anthropic.claude-3-sonnet-20240229-v1:0": {
                            "total_tokens": 4096,
                            "concurrency": 60,
                        }
                    }
                }
            ),
        ),
    ],
)
def test_config_model_limits(values: dict, expected: ConfigModelLimits):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.model_engine_limits == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigBillingEvent()),
        (
            {
                "AIGW_BILLING_EVENT__ENABLED": "yes",
                "AIGW_BILLING_EVENT__ENDPOINT": "endpoint.test",
                "AIGW_BILLING_EVENT__BATCH_SIZE": "8",
                "AIGW_BILLING_EVENT__THREAD_COUNT": "7",
            },
            ConfigBillingEvent(
                enabled=True,
                endpoint="endpoint.test",
                thread_count=7,
                batch_size=8,
            ),
        ),
        (
            {
                "AIGW_BILLING_EVENT__ENABLED": "yes",
                "AIGW_BILLING_EVENT__ENDPOINT": "endpoint.test",
            },
            ConfigBillingEvent(
                enabled=True,
                endpoint="endpoint.test",
            ),
        ),
    ],
)
def test_config_billing_event(values: dict, expected: ConfigBillingEvent):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.billing_event == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigAuditEvent()),
        (
            {
                "AIGW_AUDIT_EVENT__ENABLED": "yes",
                "AIGW_AUDIT_EVENT__BUFFER_SIZE": "50",
                "AIGW_AUDIT_EVENT__FLUSH_INTERVAL_SECONDS": "5.0",
                "AIGW_AUDIT_EVENT__MAX_RETRIES": "5",
            },
            ConfigAuditEvent(
                enabled=True,
                buffer_size=50,
                flush_interval_seconds=5.0,
                max_retries=5,
            ),
        ),
    ],
)
def test_config_audit_event(values: dict, expected: ConfigAuditEvent):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.audit_event == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, None),
        (
            {"AIGW_BEDROCK_GUARDRAIL_CONFIG": '{"guardrailIdentifier": "abc123"}'},
            ConfigBedrockGuardrail(guardrailIdentifier="abc123"),
        ),
        (
            {
                "AIGW_BEDROCK_GUARDRAIL_CONFIG": '{"guardrailIdentifier": "abc123", "guardrailVersion": "1"}'
            },
            ConfigBedrockGuardrail(
                guardrailIdentifier="abc123",
                guardrailVersion="1",
                trace="disabled",
            ),
        ),
        (
            {
                "AIGW_BEDROCK_GUARDRAIL_CONFIG": (
                    '{"guardrailIdentifier": "abc123",'
                    ' "guardrailVersion": "2", "trace": "enabled"}'
                )
            },
            ConfigBedrockGuardrail(
                guardrailIdentifier="abc123",
                guardrailVersion="2",
                trace="enabled",
            ),
        ),
        (
            {
                "AIGW_BEDROCK_GUARDRAIL_CONFIG": '{"guardrailIdentifier": "abc123", "guardrailVersion": "DRAFT"}'
            },
            ConfigBedrockGuardrail(
                guardrailIdentifier="abc123", guardrailVersion="DRAFT"
            ),
        ),
        (
            {
                "AIGW_BEDROCK_GUARDRAIL_CONFIG": '{"guardrailIdentifier": "abc123", "guardrailVersion": "12345678"}'
            },
            ConfigBedrockGuardrail(
                guardrailIdentifier="abc123", guardrailVersion="12345678"
            ),
        ),
        (
            {
                "AIGW_BEDROCK_GUARDRAIL_CONFIG": '{"guardrailIdentifier": "abc123", "guardrailVersion": ""}'
            },
            ConfigBedrockGuardrail(guardrailIdentifier="abc123", guardrailVersion=""),
        ),
        (
            {
                "AIGW_BEDROCK_GUARDRAIL_CONFIG": (
                    '{"guardrailIdentifier":'
                    ' "arn:aws:bedrock:us-east-1:123456789012:guardrail/abc123"}'
                )
            },
            ConfigBedrockGuardrail(
                guardrailIdentifier="arn:aws:bedrock:us-east-1:123456789012:guardrail/abc123",
            ),
        ),
        (
            {
                "AIGW_BEDROCK_GUARDRAIL_CONFIG": (
                    '{"guardrailIdentifier":'
                    ' "arn:aws-cn:bedrock:cn-north-1:123456789012:guardrail/def456"}'
                )
            },
            ConfigBedrockGuardrail(
                guardrailIdentifier="arn:aws-cn:bedrock:cn-north-1:123456789012:guardrail/def456",
            ),
        ),
    ],
)
def test_config_bedrock_guardrail(values: dict, expected: ConfigBedrockGuardrail):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.bedrock_guardrail_config == expected


@pytest.mark.parametrize(
    ("values", "expected_use_caching_proxy"),
    [
        ({}, False),
        ({"DUO_WORKFLOW_USE_CACHING_PROXY": "true"}, True),
        ({"DUO_WORKFLOW_USE_CACHING_PROXY": "false"}, False),
    ],
)
def test_duo_workflow_config_use_caching_proxy(
    values: dict, expected_use_caching_proxy: bool
):
    with mock.patch.dict(os.environ, values, clear=True):
        config = ConfigDuoWorkflow()

        assert config.use_caching_proxy == expected_use_caching_proxy


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigCachingProxy(url="http://localhost:8888")),
        (
            {
                "DUO_WORKFLOW_CACHING_PROXY_URL": "http://proxy.test:9999",
            },
            ConfigCachingProxy(url="http://proxy.test:9999"),
        ),
        (
            {
                "DUO_WORKFLOW_CACHING_PROXY_URL": "http://localhost:8888",
            },
            ConfigCachingProxy(url="http://localhost:8888"),
        ),
    ],
)
def test_caching_proxy_config_url(values: dict, expected: ConfigCachingProxy):
    with mock.patch.dict(os.environ, values, clear=True):
        config = ConfigCachingProxy()

        assert config.url == expected.url


def test_duo_workflow_config_defaults():
    with mock.patch.dict(os.environ, {}, clear=True):
        config = ConfigDuoWorkflow()

        assert config.use_caching_proxy is False
        assert config.caching_proxy.url == "http://localhost:8888"


def test_aigw_config_has_duo_workflow_field():
    with mock.patch.dict(
        os.environ,
        {
            "DUO_WORKFLOW_USE_CACHING_PROXY": "true",
            "DUO_WORKFLOW_CACHING_PROXY_URL": "http://test.proxy:9000",
        },
        clear=True,
    ):
        config = Config()

        duo_workflow = config.duo_workflow

        assert isinstance(duo_workflow, ConfigDuoWorkflow)
        assert duo_workflow.use_caching_proxy is True
        assert duo_workflow.caching_proxy.url == "http://test.proxy:9000"


@pytest.mark.parametrize(
    ("duo_workflow_config", "expected_url"),
    [
        (ConfigDuoWorkflow(use_caching_proxy=False), None),
        (
            ConfigDuoWorkflow(
                use_caching_proxy=True,
                caching_proxy=ConfigCachingProxy(url="http://proxy.test:8888"),
            ),
            "http://proxy.test:8888",
        ),
    ],
)
def test_caching_proxy_url(duo_workflow_config, expected_url):
    assert duo_workflow_config.caching_proxy_url() == expected_url


@pytest.mark.parametrize(
    "invalid_json",
    [
        "not-valid-json",
        "{}",
        '{"guardrailVersion": "1"}',
        '{"guardrailVersion": "1", "trace": "enabled"}',
        '{"guardrailIdentifier": ""}',
    ],
)
def test_config_bedrock_guardrail_missing_identifier(invalid_json):
    values = {"AIGW_BEDROCK_GUARDRAIL_CONFIG": invalid_json}
    with mock.patch.dict(os.environ, values, clear=True):
        with pytest.raises(ValidationError):
            Config(_env_file=None)


def test_config_bedrock_guardrail_invalid_trace_value():
    values = {
        "AIGW_BEDROCK_GUARDRAIL_CONFIG": (
            '{"guardrailIdentifier": "abc123",'
            ' "guardrailVersion": "1", "trace": "invalid"}'
        )
    }
    with mock.patch.dict(os.environ, values, clear=True):
        with pytest.raises(ValidationError):
            Config(_env_file=None)


@pytest.mark.parametrize(
    "invalid_identifier",
    [
        "HAS-UPPERCASE",
        "has spaces",
        "special!chars",
        "arn:aws:bedrock:us-east-1:short:guardrail/abc",
        "arn:aws:s3:us-east-1:123456789012:guardrail/abc",
    ],
)
def test_config_bedrock_guardrail_invalid_identifier(invalid_identifier):
    values = {
        "AIGW_BEDROCK_GUARDRAIL_CONFIG": f'{{"guardrailIdentifier": "{invalid_identifier}"}}'
    }
    with mock.patch.dict(os.environ, values, clear=True):
        with pytest.raises(ValidationError):
            Config(_env_file=None)


@pytest.mark.parametrize(
    "invalid_version",
    [
        "0",
        "01",
        "abc",
        "123456789",
        "draft",
    ],
)
def test_config_bedrock_guardrail_invalid_version(invalid_version):
    values = {
        "AIGW_BEDROCK_GUARDRAIL_CONFIG": f'{{"guardrailIdentifier": "abc123", "guardrailVersion": "{invalid_version}"}}'
    }
    with mock.patch.dict(os.environ, values, clear=True):
        with pytest.raises(ValidationError):
            Config(_env_file=None)


def test_config_bedrock_guardrail_identifier_max_length():
    long_id = "a" * 2049
    values = {
        "AIGW_BEDROCK_GUARDRAIL_CONFIG": f'{{"guardrailIdentifier": "{long_id}"}}'
    }
    with mock.patch.dict(os.environ, values, clear=True):
        with pytest.raises(ValidationError):
            Config(_env_file=None)


def test_config_bedrock_guardrail_identifier_at_max_length():
    max_id = "a" * 2048
    values = {"AIGW_BEDROCK_GUARDRAIL_CONFIG": f'{{"guardrailIdentifier": "{max_id}"}}'}
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)
        assert config.bedrock_guardrail_config.guardrailIdentifier == max_id


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigModelSelection()),
        (
            {
                "AIGW_MODEL_SELECTION__DEFAULT_MODELS": '{"duo_chat": ["claude_sonnet_4_5_20250929_vertex"]}'
            },
            ConfigModelSelection(
                default_models={"duo_chat": ["claude_sonnet_4_5_20250929_vertex"]}
            ),
        ),
        (
            {
                "AIGW_MODEL_SELECTION__DEFAULT_MODELS": (
                    '{"duo_chat": ["model_a"], "code_generations": ["model_b", "model_c"]}'
                )
            },
            ConfigModelSelection(
                default_models={
                    "duo_chat": ["model_a"],
                    "code_generations": ["model_b", "model_c"],
                }
            ),
        ),
        (
            {
                "AIGW_MODEL_SELECTION__MODEL_PARAMS": (
                    '{"claude_sonnet_4_20250514_bedrock": '
                    '{"model": "arn:aws:bedrock:us-east-2:681816819199:application-inference-profile/oitsuvtb0pij"}}'
                )
            },
            ConfigModelSelection(
                model_params={
                    "claude_sonnet_4_20250514_bedrock": {
                        "model": "arn:aws:bedrock:us-east-2:681816819199:application-inference-profile/oitsuvtb0pij"
                    }
                }
            ),
        ),
        (
            {
                "AIGW_MODEL_SELECTION__PROMPT_PARAMS": (
                    '{"claude_sonnet_4_5_20250929_vertex": {"vertex_location": "us-east5"}}'
                )
            },
            ConfigModelSelection(
                prompt_params={
                    "claude_sonnet_4_5_20250929_vertex": {"vertex_location": "us-east5"}
                }
            ),
        ),
    ],
)
def test_config_model_selection(values: dict, expected: ConfigModelSelection):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.model_selection == expected


def test_config_model_releases_not_revealed_in_repr():
    secret_payload = '{"models": [{"gitlab_identifier": "SENTINEL_SHOULD_NOT_APPEAR"}]}'
    values = {"AIGW_MODEL_SELECTION__MODEL_RELEASES": secret_payload}
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert "SENTINEL_SHOULD_NOT_APPEAR" not in repr(config)
        assert "SENTINEL_SHOULD_NOT_APPEAR" not in repr(config.model_selection)


@pytest.mark.parametrize(
    ("env", "expected_location"),
    [
        ({}, "global"),
        ({"VERTEXAI_LOCATION": "europe-west1"}, "europe-west1"),
        ({"VERTEXAI_LOCATION": ""}, "global"),
    ],
)
def test_setup_litellm_vertex_location(env: dict, expected_location: str):
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(litellm, "vertex_location", None),
        mock.patch.object(litellm, "vertex_project", None),
    ):
        setup_litellm(Config(_env_file=None))

        assert litellm.vertex_location == expected_location


@pytest.fixture
def mtls_cert_files(tmp_path):
    """Real PEM material: a CA plus client and server certificates it signed.

    The wire-level regression test completes a real TLS handshake, so the files
    must parse. The kwarg-shape tests only pass the paths through.
    """
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    def _name(cn):
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])

    def _key_usage(*, ca):
        return x509.KeyUsage(
            digital_signature=not ca,
            content_commitment=False,
            key_encipherment=False,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=ca,
            crl_sign=ca,
            encipher_only=False,
            decipher_only=False,
        )

    def _cert(cn, key, issuer_cert, issuer_key, *, ca=False, eku=None, san=None):
        now = datetime.now(timezone.utc)
        builder = (
            x509.CertificateBuilder()
            .subject_name(_name(cn))
            .issuer_name(issuer_cert.subject if issuer_cert else _name(cn))
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(hours=1))
            .add_extension(
                x509.BasicConstraints(ca=ca, path_length=None), critical=True
            )
            .add_extension(_key_usage(ca=ca), critical=True)
        )
        if eku:
            builder = builder.add_extension(x509.ExtendedKeyUsage(eku), critical=False)
        if san:
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san), critical=False
            )
        return builder.sign(issuer_key or key, hashes.SHA256())

    def _write(fname, data):
        path = tmp_path / fname
        path.write_bytes(data)
        return str(path)

    def _key_pem(key):
        return key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_cert = _cert("test-mtls-ca", ca_key, None, None, ca=True)
    client_key = ec.generate_private_key(ec.SECP256R1())
    client_cert = _cert(
        "aigw-client",
        client_key,
        ca_cert,
        ca_key,
        eku=[ExtendedKeyUsageOID.CLIENT_AUTH],
    )
    server_key = ec.generate_private_key(ec.SECP256R1())
    server_cert = _cert(
        "localhost",
        server_key,
        ca_cert,
        ca_key,
        eku=[ExtendedKeyUsageOID.SERVER_AUTH],
        san=[x509.DNSName("localhost")],
    )

    return mock.Mock(
        cert=_write("client.pem", client_cert.public_bytes(serialization.Encoding.PEM)),
        key=_write("client.key", _key_pem(client_key)),
        ca=_write("ca.crt", ca_cert.public_bytes(serialization.Encoding.PEM)),
        server_cert=_write(
            "server.pem", server_cert.public_bytes(serialization.Encoding.PEM)
        ),
        server_key=_write("server.key", _key_pem(server_key)),
    )


def test_setup_litellm_mtls_disabled():
    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.object(litellm, "aclient_session", None),
        mock.patch.object(litellm, "client_session", None),
        mock.patch("ai_gateway.config.httpx.AsyncClient") as async_client,
        mock.patch("ai_gateway.config.httpx.Client") as sync_client,
    ):
        setup_litellm(Config(_env_file=None))

        async_client.assert_not_called()
        sync_client.assert_not_called()
        assert litellm.aclient_session is None
        assert litellm.client_session is None


@pytest.mark.parametrize(
    ("include_key", "include_password", "expected"),
    [
        (False, False, "cert"),
        (True, False, "cert_key"),
        (True, True, "cert_key_password"),
    ],
)
def test_setup_litellm_mtls_resolves_cert(
    include_key, include_password, expected, mtls_cert_files
):
    env = {
        "AIGW_MTLS__ENABLED": "true",
        "AIGW_MTLS__CERT_FILE": mtls_cert_files.cert,
    }
    if include_key:
        env["AIGW_MTLS__KEY_FILE"] = mtls_cert_files.key
    if include_password:
        env["AIGW_MTLS__KEY_PASSWORD"] = "secret"

    expected_cert = {
        "cert": mtls_cert_files.cert,
        "cert_key": (mtls_cert_files.cert, mtls_cert_files.key),
        "cert_key_password": (mtls_cert_files.cert, mtls_cert_files.key, "secret"),
    }[expected]

    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(litellm, "aclient_session", None),
        mock.patch.object(litellm, "client_session", None),
        mock.patch("ai_gateway.config.httpx.AsyncClient") as async_client,
        mock.patch("ai_gateway.config.httpx.Client") as sync_client,
    ):
        setup_litellm(Config(_env_file=None))

        async_client.assert_called_once_with(cert=expected_cert, verify=True)
        sync_client.assert_called_once_with(cert=expected_cert, verify=True)
        assert litellm.aclient_session is async_client.return_value
        assert litellm.client_session is sync_client.return_value


def test_setup_litellm_mtls_verify_uses_ca_bundle(mtls_cert_files):
    env = {
        "AIGW_MTLS__ENABLED": "true",
        "AIGW_MTLS__CERT_FILE": mtls_cert_files.cert,
        "AIGW_MTLS__CA_BUNDLE": mtls_cert_files.ca,
    }
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(litellm, "aclient_session", None),
        mock.patch.object(litellm, "client_session", None),
        mock.patch("ai_gateway.config.httpx.AsyncClient") as async_client,
        mock.patch("ai_gateway.config.httpx.Client") as sync_client,
    ):
        setup_litellm(Config(_env_file=None))

        # The CA bundle must arrive as an ssl.SSLContext, never as a path: httpx
        # returns early on a string `verify` and silently drops `cert`. One
        # context per client, because httpx/httpcore mutate what they are given.
        async_client.assert_called_once()
        sync_client.assert_called_once()
        async_verify = async_client.call_args.kwargs["verify"]
        sync_verify = sync_client.call_args.kwargs["verify"]
        assert async_client.call_args.kwargs["cert"] == mtls_cert_files.cert
        assert sync_client.call_args.kwargs["cert"] == mtls_cert_files.cert
        assert isinstance(async_verify, ssl.SSLContext)
        assert isinstance(sync_verify, ssl.SSLContext)
        assert async_verify is not sync_verify


def test_setup_litellm_mtls_verify_disabled(mtls_cert_files):
    env = {
        "AIGW_MTLS__ENABLED": "true",
        "AIGW_MTLS__CERT_FILE": mtls_cert_files.cert,
        "AIGW_MTLS__VERIFY": "false",
    }
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(litellm, "aclient_session", None),
        mock.patch.object(litellm, "client_session", None),
        mock.patch("ai_gateway.config.httpx.AsyncClient") as async_client,
        mock.patch("ai_gateway.config.httpx.Client") as sync_client,
    ):
        setup_litellm(Config(_env_file=None))

        async_client.assert_called_once_with(cert=mtls_cert_files.cert, verify=False)
        sync_client.assert_called_once_with(cert=mtls_cert_files.cert, verify=False)


def test_setup_litellm_mtls_client_cert_reaches_the_wire(mtls_cert_files):
    """A CA bundle plus a client certificate must still present the certificate.

    Regression test for
    https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/2672:
    a string `verify` makes httpx 0.28 return before load_cert_chain, so the
    client certificate silently never reaches the handshake. Kwarg assertions
    cannot catch that; this test completes a real handshake against a server
    that requires a client certificate.
    """
    seen = {}
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(mtls_cert_files.server_cert, mtls_cert_files.server_key)
    server_ctx.verify_mode = ssl.CERT_REQUIRED
    server_ctx.load_verify_locations(mtls_cert_files.ca)

    def serve(srv):
        try:
            conn, _ = srv.accept()
            with server_ctx.wrap_socket(conn, server_side=True) as tls:
                seen["peer"] = tls.getpeercert()
                tls.recv(4096)
                tls.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
        except Exception as exc:  # surfaced by the handshake assertion below
            seen["error"] = exc

    env = {
        "AIGW_MTLS__ENABLED": "true",
        "AIGW_MTLS__CERT_FILE": mtls_cert_files.cert,
        "AIGW_MTLS__KEY_FILE": mtls_cert_files.key,
        "AIGW_MTLS__CA_BUNDLE": mtls_cert_files.ca,
    }
    with socket.create_server(("127.0.0.1", 0)) as srv:
        srv.settimeout(10)
        port = srv.getsockname()[1]
        thread = threading.Thread(target=serve, args=(srv,), daemon=True)
        thread.start()
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(litellm, "aclient_session", None),
            mock.patch.object(litellm, "client_session", None),
        ):
            setup_litellm(Config(_env_file=None))
            try:
                response = litellm.client_session.get(
                    f"https://localhost:{port}/", timeout=10
                )
            except Exception as client_exc:
                thread.join(timeout=10)
                raise AssertionError(
                    f"request failed ({client_exc!r}); server-side handshake "
                    f"result: {seen.get('error')!r}"
                ) from client_exc
        thread.join(timeout=10)

    assert response.status_code == 200
    subject = dict(item[0] for item in seen["peer"]["subject"])
    assert subject["commonName"] == "aigw-client"


def test_config_mtls_enabled_without_cert_raises():
    with (
        mock.patch.dict(os.environ, {"AIGW_MTLS__ENABLED": "true"}, clear=True),
        pytest.raises(ValidationError, match="cert_file must be set when mTLS"),
    ):
        Config(_env_file=None)


def test_config_mtls_enabled_with_missing_cert_file_raises():
    with (
        mock.patch.dict(
            os.environ,
            {"AIGW_MTLS__ENABLED": "true", "AIGW_MTLS__CERT_FILE": "/no/such.pem"},
            clear=True,
        ),
        pytest.raises(ValidationError, match="cert_file does not point to a file"),
    ):
        Config(_env_file=None)


def test_config_mtls_enabled_with_missing_key_file_raises(mtls_cert_files):
    with (
        mock.patch.dict(
            os.environ,
            {
                "AIGW_MTLS__ENABLED": "true",
                "AIGW_MTLS__CERT_FILE": mtls_cert_files.cert,
                "AIGW_MTLS__KEY_FILE": "/no/such.key",
            },
            clear=True,
        ),
        pytest.raises(ValidationError, match="key_file does not point to a file"),
    ):
        Config(_env_file=None)


def test_config_mtls_key_password_without_key_file_raises(mtls_cert_files):
    with (
        mock.patch.dict(
            os.environ,
            {
                "AIGW_MTLS__ENABLED": "true",
                "AIGW_MTLS__CERT_FILE": mtls_cert_files.cert,
                "AIGW_MTLS__KEY_PASSWORD": "secret",
            },
            clear=True,
        ),
        pytest.raises(ValidationError, match="key_file must be set when key_password"),
    ):
        Config(_env_file=None)


def test_config_mtls_enabled_with_missing_ca_bundle_raises(mtls_cert_files):
    with (
        mock.patch.dict(
            os.environ,
            {
                "AIGW_MTLS__ENABLED": "true",
                "AIGW_MTLS__CERT_FILE": mtls_cert_files.cert,
                "AIGW_MTLS__CA_BUNDLE": "/no/such.ca",
            },
            clear=True,
        ),
        pytest.raises(ValidationError, match="ca_bundle does not point to a file"),
    ):
        Config(_env_file=None)


def test_config_mtls_defaults_disabled():
    assert ConfigMtls().enabled is False


@pytest.mark.parametrize(
    ("url", "expected_port"),
    [
        ("http://localhost:4567", 4567),
        ("http://localhost:8888", 8888),
        ("http://example.com", 80),
        ("https://example.com", 443),
    ],
)
def test_config_mock_usage_quota_server_port_parsed_from_url(
    url: str, expected_port: int
):
    assert ConfigMockUsageQuotaServer(url=url).port == expected_port


def test_config_mock_usage_quota_server_uses_documented_env_var():
    with mock.patch.dict(
        os.environ,
        {"AIGW_MOCK_USAGE_QUOTA_SERVER__URL": "http://localhost:9999"},
        clear=True,
    ):
        config = Config(_env_file=None)

    assert config.mock_usage_quota_server.url == "http://localhost:9999"
    assert config.mock_usage_quota_server.port == 9999


class TestConfigTLS:
    def test_defaults(self):
        tls = ConfigTLS()
        assert tls.enabled is False
        assert tls.cert_file is None
        assert tls.key_file is None

    def test_enabled_with_files(self, tmp_path):
        cert = tmp_path / "server.crt"
        key = tmp_path / "server.key"
        cert.touch()
        key.touch()
        tls = ConfigTLS(enabled=True, cert_file=str(cert), key_file=str(key))
        assert tls.enabled is True
        assert tls.cert_file == str(cert)
        assert tls.key_file == str(key)

    def test_enabled_without_files_raises(self):
        with pytest.raises(
            ValidationError, match="cert_file and key_file must both be set"
        ):
            ConfigTLS(enabled=True)

    def test_cert_file_nonexistent_raises(self, tmp_path):
        key_path = tmp_path / "server.key"
        key_path.touch()
        with pytest.raises(ValidationError, match="cert_file does not point to a file"):
            ConfigTLS(
                enabled=True,
                cert_file=str(tmp_path / "missing.crt"),
                key_file=str(key_path),
            )

    def test_key_file_nonexistent_raises(self, tmp_path):
        cert_path = tmp_path / "server.cert"
        cert_path.touch()
        with pytest.raises(ValidationError, match="key_file does not point to a file"):
            ConfigTLS(
                enabled=True,
                cert_file=str(cert_path),
                key_file=str(tmp_path / "missing.key"),
            )

    def test_disabled_with_nonexistent_files_does_not_raise(self):
        tls = ConfigTLS(
            enabled=False,
            cert_file="/nonexistent/cert.pem",
            key_file="/nonexistent/key.pem",
        )
        assert tls.enabled is False


class TestConfigFastApiTLS:
    def test_tls_defaults_to_disabled(self, monkeypatch):
        # Ensure no env vars bleed in from the environment
        monkeypatch.delenv("AIGW_FASTAPI__TLS__ENABLED", raising=False)
        monkeypatch.delenv("AIGW_FASTAPI__TLS__CERT_FILE", raising=False)
        monkeypatch.delenv("AIGW_FASTAPI__TLS__KEY_FILE", raising=False)
        monkeypatch.delenv("AIGW_FASTAPI__TLS__SSL_CIPHERS", raising=False)
        cfg = ConfigFastApi()
        assert cfg.tls.enabled is False
        assert cfg.tls.cert_file is None
        assert cfg.tls.key_file is None
        assert cfg.tls.ssl_ciphers == "TLSv1"

    def test_tls_reads_from_env_via_top_level_config(self, monkeypatch, tmp_path):
        """ConfigFastApi is a plain BaseModel; TLS env vars are read via the top-level Config."""
        cert = tmp_path / "server.crt"
        key = tmp_path / "server.key"
        ciphers = "TLSv1.2"
        cert.touch()
        key.touch()
        monkeypatch.setenv("AIGW_FASTAPI__TLS__ENABLED", "true")
        monkeypatch.setenv("AIGW_FASTAPI__TLS__CERT_FILE", str(cert))
        monkeypatch.setenv("AIGW_FASTAPI__TLS__KEY_FILE", str(key))
        monkeypatch.setenv("AIGW_FASTAPI__TLS__SSL_CIPHERS", str(ciphers))
        cfg = Config(_env_file=None)
        assert cfg.fastapi.tls.enabled is True
        assert cfg.fastapi.tls.cert_file == str(cert)
        assert cfg.fastapi.tls.key_file == str(key)
        assert cfg.fastapi.tls.ssl_ciphers == str(ciphers)


class TestConfigDuoWorkflowTLS:
    def test_tls_defaults_to_disabled(self, monkeypatch):
        # Ensure no env vars bleed in from the environment
        monkeypatch.delenv("DUO_WORKFLOW_TLS__ENABLED", raising=False)
        monkeypatch.delenv("DUO_WORKFLOW_TLS__CERT_FILE", raising=False)
        monkeypatch.delenv("DUO_WORKFLOW_TLS__KEY_FILE", raising=False)
        cfg = ConfigDuoWorkflow()
        assert cfg.tls.enabled is False
        assert cfg.tls.cert_file is None
        assert cfg.tls.key_file is None

    def test_tls_reads_from_env(self, monkeypatch, tmp_path):
        cert = tmp_path / "server.crt"
        key = tmp_path / "server.key"
        cert.touch()
        key.touch()
        monkeypatch.setenv("DUO_WORKFLOW_TLS__ENABLED", "true")
        monkeypatch.setenv("DUO_WORKFLOW_TLS__CERT_FILE", str(cert))
        monkeypatch.setenv("DUO_WORKFLOW_TLS__KEY_FILE", str(key))
        cfg = ConfigDuoWorkflow()
        assert cfg.tls.enabled is True
        assert cfg.tls.cert_file == str(cert)
        assert cfg.tls.key_file == str(key)


class TestConfigDuoChat:
    def test_model_request_timeout_default(self):
        """ConfigDuoChat.model_request_timeout defaults to 30.0 seconds."""
        cfg = ConfigDuoChat()
        assert cfg.model_request_timeout == 30.0

    def test_model_request_timeout_reads_from_env_var(self, monkeypatch):
        """AIGW_DUO_CHAT__MODEL_REQUEST_TIMEOUT overrides the default."""
        monkeypatch.setenv("AIGW_DUO_CHAT__MODEL_REQUEST_TIMEOUT", "120.0")
        config = Config(_env_file=None)
        assert config.duo_chat.model_request_timeout == 120.0

    def test_model_request_timeout_must_be_positive(self):
        """model_request_timeout rejects non-positive values."""
        with pytest.raises(ValidationError):
            ConfigDuoChat(model_request_timeout=0.0)

        with pytest.raises(ValidationError):
            ConfigDuoChat(model_request_timeout=-1.0)
