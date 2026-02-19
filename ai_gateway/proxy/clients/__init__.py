# flake8: noqa

from typing import Any

from litellm.llms.openai.openai import OpenAIChatCompletionResponseIterator
from litellm.proxy.pass_through_endpoints import pass_through_endpoints
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.utils import ModelResponseStream

from ai_gateway.proxy.clients.anthropic import AnthropicProxyModelFactory
from ai_gateway.proxy.clients.base import ProxyClient, ProxyModel
from ai_gateway.proxy.clients.openai import OpenAIProxyModelFactory
from ai_gateway.proxy.clients.token_usage import TokenUsage
from ai_gateway.proxy.clients.vertex_ai import VertexAIProxyModelFactory


# Monkey-patches to support calling litellm-proxy's code without it handling fastapi routing
def _is_registered_pass_through_route(
    route: str,
) -> bool:  # pylint: disable=unused-argument
    return True


pass_through_endpoints.InitPassThroughEndpointHelpers.is_registered_pass_through_route = (
    _is_registered_pass_through_route
)


def _get_registered_pass_through_route(
    route: str,
) -> dict[str, Any]:  # pylint: disable=unused-argument
    return {}


pass_through_endpoints.InitPassThroughEndpointHelpers.get_registered_pass_through_route = (
    _get_registered_pass_through_route
)


def _chunk_parser(self, chunk: dict) -> ModelResponseStream:
    """
    Monkey-patched version of the upstream parser
    (https://github.com/BerriAI/litellm/blob/v1.81.0-stable/litellm/llms/openai/openai.py#L328-L335)
    that adds the minimal fields needed to track proxy usage (`model` and `usage`) if missing.
    """
    try:
        kwargs = {**chunk}

        # This is our patched-in code. Wrap it in a `try` block that discards any exception originating from it, so the
        # original functionality is unaffected by any error of ours.
        try:
            if "response" in chunk:
                if "model" in chunk["response"] and "model" not in kwargs:
                    kwargs["model"] = chunk["response"]["model"]

                if "usage" in chunk["response"] and "usage" not in kwargs:
                    kwargs["usage"] = (
                        ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
                            chunk["response"]["usage"]
                        )
                    )
        except Exception:
            pass

        return ModelResponseStream(**kwargs)
    except Exception as e:
        raise e


OpenAIChatCompletionResponseIterator.chunk_parser = _chunk_parser
