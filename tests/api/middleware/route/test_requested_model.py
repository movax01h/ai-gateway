import pytest
from pydantic import Field
from starlette_context import context as starlette_context
from starlette_context import request_cycle_context

from ai_gateway.api.middleware.route import record_requested_model
from ai_gateway.api.v2.code.typing import CurrentFile, SuggestionsRequest
from ai_gateway.models import KindModelProvider


class _TestSuggestionsRequest(SuggestionsRequest):
    current_file: CurrentFile = Field(
        default_factory=lambda: CurrentFile(
            file_name="test.py",
            content_above_cursor="",
            content_below_cursor="",
        )
    )


class _Payload(_TestSuggestionsRequest):
    pass


class _PayloadWithUnsetModelFields(_TestSuggestionsRequest):
    pass


@pytest.fixture(name="request_context", autouse=True)
def request_context_fixture():
    with request_cycle_context({}):
        yield starlette_context


class TestRecordRequestedModel:
    @pytest.mark.asyncio
    async def test_records_requested_model_provider_and_name(self, request_context):
        @record_requested_model()
        async def endpoint(*, payload):
            return "ok"

        await endpoint(
            payload=_Payload(
                model_provider=KindModelProvider.VERTEX_AI, model_name="foo"
            )
        )

        assert request_context["meta.requested_model_provider"] == "vertex-ai"
        assert request_context["meta.requested_model_name"] == "foo"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("payload", "expected_provider", "expected_name"),
        [
            (_Payload(), None, None),
            (_Payload(model_provider=KindModelProvider.VERTEX_AI), "vertex-ai", None),
            (_Payload(model_name="foo"), None, "foo"),
            (_PayloadWithUnsetModelFields(), None, None),
        ],
    )
    async def test_records_none_for_omitted_model_fields(
        self, request_context, payload, expected_provider, expected_name
    ):
        @record_requested_model()
        async def endpoint(*, payload):
            return "ok"

        await endpoint(payload=payload)

        assert request_context["meta.requested_model_provider"] == expected_provider
        assert request_context["meta.requested_model_name"] == expected_name

    @pytest.mark.asyncio
    async def test_executes_wrapped_endpoint_and_returns_its_result(self):
        calls = []

        @record_requested_model()
        async def endpoint(request, *, payload, extra):
            calls.append((request, payload, extra))
            return "result"

        payload = _Payload(model_name="foo")

        assert await endpoint("request", payload=payload, extra=1) == "result"
        assert calls == [("request", payload, 1)]

    def test_preserves_wrapped_function_metadata(self):
        async def endpoint(*, payload):
            """Endpoint docstring."""

        wrapped = record_requested_model()(endpoint)

        assert wrapped.__name__ == "endpoint"
        assert wrapped.__doc__ == "Endpoint docstring."
        assert wrapped.__wrapped__ is endpoint

    @pytest.mark.asyncio
    async def test_uses_payload_keyword_when_other_pydantic_argument_is_present(
        self, request_context
    ):
        @record_requested_model()
        async def endpoint(*, payload, other):
            return "ok"

        await endpoint(
            payload=_Payload(
                model_provider=KindModelProvider.VERTEX_AI, model_name="requested"
            ),
            other=_Payload(model_provider="anthropic", model_name="other-model"),
        )

        assert request_context["meta.requested_model_provider"] == "vertex-ai"
        assert request_context["meta.requested_model_name"] == "requested"

    @pytest.mark.asyncio
    async def test_ignores_other_pydantic_argument_when_payload_is_invalid(
        self, request_context
    ):
        @record_requested_model()
        async def endpoint(*, payload, other):
            return "ok"

        await endpoint(
            payload="not a pydantic model",
            other=_Payload(model_provider="anthropic", model_name="other-model"),
        )

        assert "meta.requested_model_provider" not in request_context
        assert "meta.requested_model_name" not in request_context

    @pytest.mark.asyncio
    async def test_runs_endpoint_without_recording_when_there_is_no_payload(
        self, request_context
    ):
        @record_requested_model()
        async def endpoint(*, other):
            return "ok"

        assert await endpoint(other="not a payload") == "ok"
        assert "meta.requested_model_provider" not in request_context
        assert "meta.requested_model_name" not in request_context

    @pytest.mark.asyncio
    async def test_records_before_the_wrapped_decorator_runs(self, request_context):
        seen = {}

        def inner_decorator(func):
            async def wrapper(*args, **kwargs):
                seen["name"] = starlette_context.get("meta.requested_model_name")
                return "denied"

            return wrapper

        @record_requested_model()
        @inner_decorator
        async def endpoint(*, payload):
            return "ok"

        assert await endpoint(payload=_Payload(model_name="foo")) == "denied"
        assert seen == {"name": "foo"}
