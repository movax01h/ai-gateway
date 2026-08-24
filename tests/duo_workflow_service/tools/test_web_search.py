import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.tools import ToolException

from ai_gateway.model_selection.models import ModelClassProvider
from duo_workflow_service.tools.web_search import (
    _MCP_PROTOCOL_VERSION,
    AgentCoreWebSearch,
    WebSearchInput,
    WebSearchSettings,
    _build_domain_filter,
    _extract_mcp_payload,
    _negotiated_protocol_version,
    _parse_results,
    native_web_search_available,
    web_search_boto_session,
)
from lib.context import current_model_metadata_context
from lib.feature_flags.context import FeatureFlag, current_feature_flag_context

GATEWAY_URL = (
    "https://gw-abcdefghij.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
)


@pytest.fixture(name="isolate_web_search_env", autouse=True)
def isolate_web_search_env_fixture(monkeypatch):
    """Keep settings read from the environment out of these tests.

    The test process loads the repository `.env`, so a developer who has pointed the tool at a real
    gateway would otherwise be testing against those values: anything asserting the unconfigured
    behaviour would pass in CI and fail locally, or vice versa.
    """
    prefix = WebSearchSettings.model_config["env_prefix"]
    for field in WebSearchSettings.model_fields:
        monkeypatch.delenv(f"{prefix}{field.upper()}", raising=False)


def _tool_result(results: list[dict]) -> dict:
    """Wrap results the way the connector does: a JSON string inside a content block."""
    return {
        "result": {
            "isError": False,
            "content": [
                {"type": "text", "text": json.dumps({"id": "abc", "results": results})}
            ],
        }
    }


@pytest.fixture(name="settings")
def settings_fixture():
    return WebSearchSettings(
        enabled=True,
        gateway_url=GATEWAY_URL,
        target_name="web-search-tool",
        region="us-east-1",
    )


@pytest.fixture(name="metadata")
def metadata_fixture():
    credentials = MagicMock()
    credentials.get_frozen_credentials.return_value = MagicMock(
        access_key="AKIA", secret_key="secret", token=None
    )
    session = MagicMock()
    session.get_credentials.return_value = credentials
    return {"boto_session": session}


@pytest.fixture(name="tool")
def tool_fixture(settings, metadata):
    tool = AgentCoreWebSearch(metadata=metadata)
    tool._settings = settings
    return tool


@pytest.fixture(name="mock_post")
def mock_post_fixture():
    """Patch httpx so initialize, initialized and tools/call all resolve."""

    def _make(final_payload: dict, status_code: int = 200):
        def _response(payload: dict) -> MagicMock:
            response = MagicMock(spec=httpx.Response)
            response.status_code = status_code
            response.headers = {
                "content-type": "application/json",
                "Mcp-Session-Id": "sess-1",
            }
            response.json.return_value = payload
            return response

        client = AsyncMock()
        client.post = AsyncMock(
            side_effect=[
                _response({"result": {}}),
                _response({}),
                _response(final_payload),
            ]
        )
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        return patch("httpx.AsyncClient", return_value=ctx), client

    return _make


class TestDomainFilter:
    @pytest.mark.parametrize(
        "allowed,blocked,expected",
        [
            (None, None, None),
            ([], [], None),
            (["a.com"], None, {"include": ["a.com"]}),
            (None, ["b.com"], {"exclude": ["b.com"]}),
            (["a.com"], ["b.com"], {"include": ["a.com"], "exclude": ["b.com"]}),
        ],
    )
    def test_build_domain_filter(self, allowed, blocked, expected):
        assert _build_domain_filter(allowed, blocked) == expected

    @pytest.mark.parametrize(
        "field,key", [("allowed", "include"), ("blocked", "exclude")]
    )
    def test_caps_at_100_domains(self, field, key):
        domains = [f"d{i}.com" for i in range(150)]
        result = _build_domain_filter(
            domains if field == "allowed" else None,
            domains if field == "blocked" else None,
        )
        assert len(result[key]) == 100


class TestPayloadParsing:
    def test_parses_results(self):
        results = _parse_results(
            _tool_result(
                [
                    {
                        "text": "Python 3.13 released",
                        "url": "https://example.com/py",
                        "title": "Release",
                        "publishedDate": "2024-10-07",
                    }
                ]
            )["result"]
        )
        assert len(results) == 1
        assert results[0].url == "https://example.com/py"
        assert results[0].published_date == "2024-10-07"

    def test_tolerates_missing_optional_fields(self):
        """Only `text` is guaranteed by the connector."""
        results = _parse_results(_tool_result([{"text": "snippet only"}])["result"])
        assert results[0].text == "snippet only"
        assert results[0].url is None
        assert results[0].title is None

    @pytest.mark.parametrize(
        "payload", [{"content": []}, {"content": [{"text": '{"results": []}'}]}]
    )
    def test_empty_results(self, payload):
        assert _parse_results(payload) == []

    def test_raises_on_error_flag(self):
        with pytest.raises(ToolException, match="Web search failed"):
            _parse_results({"isError": True, "content": [{"text": "boom"}]})

    def test_raises_on_malformed_json(self):
        with pytest.raises(ToolException, match="malformed payload"):
            _parse_results({"content": [{"text": "not json"}]})

    def test_extracts_sse_payload(self):
        response = MagicMock(spec=httpx.Response)
        response.headers = {"content-type": "text/event-stream"}
        response.text = 'event: message\ndata: {"result": {"ok": true}}\n\n'
        assert _extract_mcp_payload(response) == {"result": {"ok": True}}

    def test_raises_on_sse_without_data(self):
        response = MagicMock(spec=httpx.Response)
        response.headers = {"content-type": "text/event-stream"}
        response.text = "event: ping\n\n"
        with pytest.raises(ToolException, match="no data frame"):
            _extract_mcp_payload(response)


class TestQualifiedToolName:
    def test_uses_triple_underscore_target_prefix(self, settings):
        assert settings.qualified_tool_name() == "web-search-tool___WebSearch"


class TestExecute:
    @pytest.mark.asyncio
    async def test_returns_results(self, tool, mock_post):
        patcher, client = mock_post(
            _tool_result([{"text": "hit", "url": "https://a.com"}])
        )
        with patcher:
            result = await tool._execute(query="python 3.13 release date")

        assert json.loads(result)["results"][0]["url"] == "https://a.com"
        # initialize, notifications/initialized, tools/call
        assert client.post.await_count == 3

    @pytest.mark.asyncio
    async def test_sends_domain_filter_from_metadata_not_model(self, tool, mock_post):
        """The allowlist is governance: it comes from metadata, never from tool arguments."""
        tool.metadata["web_search_allowed_domains"] = ["nvd.nist.gov"]
        tool.metadata["web_search_blocked_domains"] = ["evil.example"]
        patcher, client = mock_post(_tool_result([]))

        with patcher:
            await tool._execute(query="CVE-2025-50182")

        body = json.loads(client.post.await_args_list[-1].kwargs["content"])
        assert body["params"]["arguments"]["filters"] == {
            "domainFilter": {"include": ["nvd.nist.gov"], "exclude": ["evil.example"]}
        }
        assert body["params"]["name"] == "web-search-tool___WebSearch"

    @pytest.mark.asyncio
    async def test_omits_filters_when_no_policy(self, tool, mock_post):
        patcher, client = mock_post(_tool_result([]))
        with patcher:
            await tool._execute(query="anything")

        body = json.loads(client.post.await_args_list[-1].kwargs["content"])
        assert "filters" not in body["params"]["arguments"]

    @pytest.mark.asyncio
    async def test_forwards_max_results(self, tool, mock_post):
        patcher, client = mock_post(_tool_result([]))
        with patcher:
            await tool._execute(query="q", max_results=25)

        body = json.loads(client.post.await_args_list[-1].kwargs["content"])
        assert body["params"]["arguments"]["maxResults"] == 25

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("", "must not be empty"),
            ("   ", "must not be empty"),
            ("x" * 201, "at most 200"),
        ],
    )
    async def test_rejects_invalid_query(self, tool, query, expected):
        with pytest.raises(ToolException, match=expected):
            await tool._execute(query=query)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "overrides,expected",
        [
            ({"enabled": False}, "not enabled"),
            ({"gateway_url": ""}, "not configured"),
            ({"target_name": ""}, "not configured"),
        ],
    )
    async def test_requires_configuration(self, tool, settings, overrides, expected):
        for key, value in overrides.items():
            setattr(settings, key, value)
        with pytest.raises(ToolException, match=expected):
            await tool._execute(query="q")

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self, tool, mock_post):
        patcher, _ = mock_post(_tool_result([]), status_code=403)
        with patcher, pytest.raises(ToolException, match="HTTP 403"):
            await tool._execute(query="q")

    @pytest.mark.asyncio
    async def test_raises_on_jsonrpc_error(self, tool, mock_post):
        patcher, _ = mock_post({"error": {"code": -32602, "message": "bad params"}})
        with patcher, pytest.raises(ToolException, match="bad params"):
            await tool._execute(query="q")

    @pytest.mark.asyncio
    async def test_raises_without_aws_session(self, tool, mock_post):
        tool.metadata = {}
        patcher, _ = mock_post(_tool_result([]))
        with patcher, pytest.raises(ToolException, match="missing AWS session"):
            await tool._execute(query="q")

    @pytest.mark.asyncio
    async def test_signs_request_with_sigv4(self, tool, mock_post):
        patcher, client = mock_post(_tool_result([]))
        with patcher:
            await tool._execute(query="q")

        headers = client.post.await_args_list[-1].kwargs["headers"]
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "bedrock-agentcore" in headers["Authorization"]
        assert headers["MCP-Protocol-Version"] == _MCP_PROTOCOL_VERSION


class TestDisplayMessage:
    def test_format_display_message(self, tool):
        message = tool.format_display_message(WebSearchInput(query="urllib3 CVE"))
        assert message == "Searched the web for: urllib3 CVE"


@pytest.fixture(name="model_class_provider")
def model_class_provider_fixture(request):
    """Set the resolved model's class provider, or None for no resolved model."""
    provider = getattr(request, "param", None)
    if provider is None:
        token = current_model_metadata_context.set(None)
    else:
        metadata = MagicMock()
        metadata.llm_definition.model_class_provider = provider
        token = current_model_metadata_context.set(metadata)
    yield provider
    current_model_metadata_context.reset(token)


@pytest.fixture(name="web_search_flag")
def web_search_flag_fixture(request):
    enabled = getattr(request, "param", True)
    flags = {FeatureFlag.DAP_WEB_SEARCH.value} if enabled else set()
    token = current_feature_flag_context.set(flags)
    yield enabled
    current_feature_flag_context.reset(token)


class TestNativeWebSearchAvailable:
    @pytest.mark.parametrize(
        "model_class_provider,expected",
        [
            (ModelClassProvider.ANTHROPIC, True),
            (ModelClassProvider.OPENAI, True),
            (ModelClassProvider.LITE_LLM, False),
            (ModelClassProvider.GOOGLE_GENAI, False),
            (ModelClassProvider.AMAZON_Q, False),
        ],
        indirect=["model_class_provider"],
    )
    def test_provider_determines_native_support(self, model_class_provider, expected):
        assert native_web_search_available() is expected

    @pytest.mark.parametrize("model_class_provider", [None], indirect=True)
    def test_assumes_native_when_no_model_resolved(self, model_class_provider):
        """Conservative default: risk no search rather than a duplicated, billable one."""
        assert native_web_search_available() is True


class TestProtocolVersionNegotiation:
    @staticmethod
    def _response(payload):
        return httpx.Response(
            200, json=payload, headers={"content-type": "application/json"}
        )

    def test_uses_version_named_by_server(self):
        """A live AgentCore gateway accepts an `initialize` offering a version it does not support.

        It answers with its own version and only rejects the mismatch on the next request, so the server's choice has to
        be echoed rather than assumed.
        """
        response = self._response({"result": {"protocolVersion": "2030-01-01"}})

        assert _negotiated_protocol_version(response) == "2030-01-01"

    @pytest.mark.parametrize(
        "payload",
        [
            {"result": {}},
            {"result": {"protocolVersion": ""}},
            {"result": {"protocolVersion": None}},
            {"result": {"protocolVersion": 20250326}},
            {},
        ],
    )
    def test_falls_back_when_server_gives_no_usable_version(self, payload):
        response = self._response(payload)

        assert _negotiated_protocol_version(response) == _MCP_PROTOCOL_VERSION

    def test_falls_back_on_unparseable_body(self):
        response = httpx.Response(
            200, content=b"not json", headers={"content-type": "application/json"}
        )

        assert _negotiated_protocol_version(response) == _MCP_PROTOCOL_VERSION

    @pytest.mark.asyncio
    async def test_negotiated_version_is_sent_on_later_requests(self, tool, mock_post):
        patcher, client = mock_post(_tool_result([]))
        with patcher:
            client.post.side_effect = None
            client.post.return_value = self._response(
                {"result": {"protocolVersion": "2030-01-01"}}
            )
            await tool._execute(query="q")

        versions = [
            call.kwargs["headers"]["MCP-Protocol-Version"]
            for call in client.post.await_args_list
        ]
        # The opening offer, then the server's choice for every subsequent request.
        assert versions[0] == _MCP_PROTOCOL_VERSION
        assert versions[1:] == ["2030-01-01"] * (len(versions) - 1)


class TestBotoSession:
    @pytest.fixture(autouse=True)
    def clear_session_cache(self):
        """The factory is cached for reuse across workflows, so each case needs a cold cache."""
        web_search_boto_session.cache_clear()
        yield
        web_search_boto_session.cache_clear()

    @pytest.mark.parametrize("enabled", [True, False])
    def test_session_built_only_when_enabled(self, enabled):
        with patch(
            "duo_workflow_service.tools.web_search.WebSearchSettings"
        ) as mock_settings:
            mock_settings.return_value.enabled = enabled

            session = web_search_boto_session()

        assert (session is not None) is enabled

    def test_session_is_reused(self):
        with patch(
            "duo_workflow_service.tools.web_search.WebSearchSettings"
        ) as mock_settings:
            mock_settings.return_value.enabled = True

            assert web_search_boto_session() is web_search_boto_session()

    def test_credentials_come_from_default_provider_chain(self):
        """Bedrock models are already credentialed this way, so web search inherits the same chain."""
        with patch(
            "duo_workflow_service.tools.web_search.WebSearchSettings"
        ) as mock_settings:
            mock_settings.return_value.enabled = True

            with patch(
                "duo_workflow_service.tools.web_search.boto3.Session"
            ) as mock_session:
                web_search_boto_session()

        mock_session.assert_called_once_with()


class TestIsAvailable:
    @pytest.mark.parametrize(
        "web_search_flag,model_class_provider,expected",
        [
            # Flag on and native unavailable: this tool is the fallback.
            (True, ModelClassProvider.LITE_LLM, True),
            (True, ModelClassProvider.GOOGLE_GENAI, True),
            # Native handles it, so the fallback stays out of the toolset.
            (True, ModelClassProvider.ANTHROPIC, False),
            (True, ModelClassProvider.OPENAI, False),
            # Flag off: never enabled, regardless of model.
            (False, ModelClassProvider.LITE_LLM, False),
            (False, ModelClassProvider.ANTHROPIC, False),
        ],
        indirect=["web_search_flag", "model_class_provider"],
    )
    def test_gating(self, web_search_flag, model_class_provider, expected, monkeypatch):
        monkeypatch.setenv("DUO_WORKFLOW_WEB_SEARCH__ENABLED", "true")
        monkeypatch.setenv("DUO_WORKFLOW_WEB_SEARCH__GATEWAY_URL", GATEWAY_URL)
        monkeypatch.setenv("DUO_WORKFLOW_WEB_SEARCH__TARGET_NAME", "websearch")

        assert AgentCoreWebSearch.is_available() is expected

    @pytest.mark.parametrize(
        "env,expected",
        [
            (
                {
                    "ENABLED": "true",
                    "GATEWAY_URL": GATEWAY_URL,
                    "TARGET_NAME": "websearch",
                },
                True,
            ),
            # Nothing set: the state every deployment starts in.
            ({}, False),
            # Switched off, however complete the rest of the configuration is.
            (
                {
                    "ENABLED": "false",
                    "GATEWAY_URL": GATEWAY_URL,
                    "TARGET_NAME": "websearch",
                },
                False,
            ),
            # Half-configured: enabling without a gateway to call must not offer the tool.
            ({"ENABLED": "true", "TARGET_NAME": "websearch"}, False),
            ({"ENABLED": "true", "GATEWAY_URL": GATEWAY_URL}, False),
        ],
        ids=["configured", "unset", "disabled", "no_gateway_url", "no_target_name"],
    )
    @pytest.mark.parametrize(
        "model_class_provider", [ModelClassProvider.LITE_LLM], indirect=True
    )
    def test_requires_configured_gateway(
        self, web_search_flag, model_class_provider, env, expected, monkeypatch
    ):
        """An unconfigured deployment must not advertise the tool.

        This is what keeps the feature dormant while the flag rolls out: offering it without a
        gateway would fail every call the model made, rather than leaving the model unaware of it.
        """
        for key, value in env.items():
            monkeypatch.setenv(f"DUO_WORKFLOW_WEB_SEARCH__{key}", value)

        assert AgentCoreWebSearch.is_available() is expected


class TestSigning:
    @pytest.mark.parametrize(
        "session,expected",
        [
            (None, "missing AWS session"),
            # A session exists but resolves to nothing: the deployment set no credentials at all, or
            # the provider chain came up empty.
            ("empty", "missing AWS credentials"),
        ],
        ids=["no_session", "no_credentials"],
    )
    def test_unusable_credentials_raise_tool_exception(
        self, settings, session, expected
    ):
        """Signing cannot proceed without credentials, and must say which part is missing.

        `is_available()` keeps an unconfigured deployment from reaching this, so hitting it means the settings and the
        AWS environment disagree — worth an error that names the difference.
        """
        metadata = {}
        if session == "empty":
            boto_session = MagicMock()
            boto_session.get_credentials.return_value = None
            metadata = {"boto_session": boto_session}

        tool = AgentCoreWebSearch(metadata=metadata)
        tool._settings = settings

        with pytest.raises(ToolException, match=expected):
            tool._credentials()

    @pytest.mark.asyncio
    async def test_credentials_resolved_off_the_event_loop(self, tool):
        """Credential resolution must not run inline on the event loop.

        The provider chain can make a blocking network call (assume-role, or a refresh near expiry), which would stall
        every other coroutine in the process for its duration.
        """
        with patch(
            "duo_workflow_service.tools.web_search.asyncio.to_thread",
            new=AsyncMock(side_effect=lambda func, *args: func(*args)),
        ) as mock_to_thread:
            await tool._signed_headers("{}", None)

        mock_to_thread.assert_awaited_once_with(tool._credentials)

    @pytest.mark.asyncio
    async def test_signature_is_computed_over_the_body(self, tool):
        """Each request needs its own signature, which is why the MCP SDK's transport is unusable."""
        first = await tool._signed_headers('{"method":"initialize"}', None)
        second = await tool._signed_headers('{"method":"tools/call"}', None)

        assert first["Authorization"] != second["Authorization"]
        assert "AWS4-HMAC-SHA256" in first["Authorization"]


class TestTransportErrors:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            httpx.ConnectTimeout("timed out"),
            httpx.ReadTimeout("timed out"),
            httpx.ConnectError("refused"),
            httpx.RemoteProtocolError("disconnected"),
        ],
        ids=lambda error: type(error).__name__,
    )
    async def test_transport_failure_raises_tool_exception(self, tool, error):
        """A single-region gateway called from every other region will time out sometimes.

        Those failures have to reach the agent as `ToolException`, like every other error this tool
        raises; a raw httpx exception escapes the tool's contract and leaks internals to the caller.
        """
        client = AsyncMock()
        client.post = AsyncMock(side_effect=error)
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client

        with patch("httpx.AsyncClient", return_value=ctx):
            with pytest.raises(ToolException) as exc_info:
                await tool._execute("gitlab ci syntax")

        assert type(error).__name__ in str(exc_info.value)
        assert "gitlab ci syntax" not in str(exc_info.value)


class TestLogging:
    """These lines are how an operator confirms a search went to AgentCore."""

    @pytest.fixture(name="captured_log")
    def captured_log_fixture(self):
        with patch("duo_workflow_service.tools.web_search.log") as log:
            yield log

    @staticmethod
    def _entry(captured_log, event: str):
        return next(
            call for call in captured_log.info.call_args_list if call.args[0] == event
        )

    @pytest.mark.asyncio
    async def test_logs_gateway_request(self, tool, mock_post, captured_log):
        patcher, _ = mock_post(_tool_result([{"text": "hit"}]))

        with patcher:
            await tool._execute(query="latest rails version", max_results=5)

        assert self._entry(
            captured_log, "Calling AgentCore web search gateway"
        ).kwargs == {
            "gateway_host": "gw-abcdefghij.gateway.bedrock-agentcore.us-east-1.amazonaws.com",
            "gateway_tool": "web-search-tool___WebSearch",
            "region": "us-east-1",
            "max_results": 5,
            "domain_filtered": False,
        }

    @pytest.mark.asyncio
    async def test_logs_completion_with_duration(self, tool, mock_post, captured_log):
        patcher, _ = mock_post(_tool_result([{"text": "a"}, {"text": "b"}]))

        with patcher:
            await tool._execute(query="latest rails version")

        entry = self._entry(captured_log, "AgentCore web search completed")
        assert entry.kwargs["result_count"] == 2
        assert entry.kwargs["domain_filtered"] is False
        assert entry.kwargs["duration_s"] >= 0

    @pytest.mark.asyncio
    async def test_reports_domain_filtering_on_both_lines(
        self, tool, mock_post, captured_log
    ):
        """Whether governance actually applied is the point of the field, so both lines carry it."""
        tool.metadata["web_search_allowed_domains"] = ["docs.gitlab.com"]
        patcher, _ = mock_post(_tool_result([]))

        with patcher:
            await tool._execute(query="ci rules keyword")

        assert all(
            call.kwargs["domain_filtered"] for call in captured_log.info.call_args_list
        )

    @pytest.mark.asyncio
    async def test_does_not_log_the_query(self, tool, mock_post, captured_log):
        """Queries are user content, so they belong in the UI message, not in logs."""
        patcher, _ = mock_post(_tool_result([]))

        with patcher:
            await tool._execute(query="unreleased codename acquisition")

        assert "unreleased codename acquisition" not in str(captured_log.mock_calls), (
            "search query text leaked into the logs"
        )
