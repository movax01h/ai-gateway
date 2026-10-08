"""Fixtures for the litellm parameter tests.

The harness is built once per session. It sets fake credentials, replaces the HTTP transport, blocks real sockets, and
builds the same prompt registry the service uses. Each test only resets what the fake network recorded.

Set LITELLM_LOCAL_MODEL_COST_MAP=True before pytest starts, as `make test-litellm-params` does. litellm reads it at
import time, which happens before any fixture runs.
"""

from dataclasses import dataclass
from typing import Any, Iterator

import litellm
import pytest

from ai_gateway.config import Config, setup_litellm
from ai_gateway.container import ContainerApplication
from litellm_param_tests import fakenet

# Importing this module registers the built-in and Fireworks models with litellm, as the service does at startup.
import ai_gateway.models.v2.chat_litellm  # noqa: F401  # isort: skip  # pylint: disable=unused-import,wrong-import-order


@dataclass
class Harness:
    """What the matrix needs to build and call a model the way production does."""

    managed_config: Config
    managed_registry: Any
    selfhosted_config: Config
    selfhosted_registry: Any
    net: fakenet.FakeNet


def _build_registry(custom_models_enabled: bool) -> tuple[Config, Any]:
    config = Config(
        _env_file=None,
        _env_prefix="AIGW_TEST",
        custom_models={"enabled": custom_models_enabled},
        google_cloud_platform={"project": "fake-project"},
    )
    assert config.mock_model_responses is False, (
        "the mock flag would replace the model classes with fakes"
    )

    setup_litellm(config)  # the same global setup the service runs at startup

    container = ContainerApplication()
    container.config.from_dict(config.model_dump())
    return config, container.pkg_prompts().prompt_registry()


def _stub_vertex_auth(mp: pytest.MonkeyPatch) -> None:
    """Skip the Google token exchange, which litellm does outside httpx.

    This replaces two private litellm methods. If litellm renames them, this raises an AttributeError. That is a signal
    to review the Vertex auth path.
    """
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

    def project_of(args: tuple, kwargs: dict) -> str:
        return (
            kwargs.get("project_id", args[1] if len(args) > 1 else None)
            or "fake-project"
        )

    async def token_async(_self: Any, *args: Any, **kwargs: Any) -> tuple[str, str]:
        return "fake-token", project_of(args, kwargs)

    def token_sync(_self: Any, *args: Any, **kwargs: Any) -> tuple[str, str]:
        return "fake-token", project_of(args, kwargs)

    mp.setattr(VertexBase, "_ensure_access_token_async", token_async)
    mp.setattr(VertexBase, "_ensure_access_token", token_sync)


@pytest.fixture(name="harness", scope="session")
def harness_fixture() -> Iterator[Harness]:
    net = fakenet.FakeNet()

    with pytest.MonkeyPatch.context() as mp:
        for name, value in {
            "AWS_ACCESS_KEY_ID": "fake",
            "AWS_SECRET_ACCESS_KEY": "fake",
            "AWS_REGION_NAME": "us-east-1",
            "AWS_REGION": "us-east-1",
            "OPENAI_API_KEY": "fake",
            "AWS_CONFIG_FILE": "/dev/null",
            "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
            "AWS_EC2_METADATA_DISABLED": "true",
        }.items():
            mp.setenv(name, value)
        for name in (
            "AWS_PROFILE",
            "AWS_BEARER_TOKEN_BEDROCK",
            "BEDROCK_MANTLE_API_KEY",
        ):
            mp.delenv(name, raising=False)

        # setup_litellm assigns these two module globals. Record the originals so they are restored.
        mp.setattr(litellm, "vertex_project", litellm.vertex_project)
        mp.setattr(litellm, "vertex_location", litellm.vertex_location)

        fakenet.install(mp, net)
        _stub_vertex_auth(mp)

        managed_config, managed_registry = _build_registry(custom_models_enabled=False)
        selfhosted_config, selfhosted_registry = _build_registry(
            custom_models_enabled=True
        )

        yield Harness(
            managed_config,
            managed_registry,
            selfhosted_config,
            selfhosted_registry,
            net,
        )

    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.fixture(name="net", autouse=True)
def net_fixture(harness: Harness) -> Iterator[fakenet.FakeNet]:
    """Give each test a clean record of what litellm sent, and drop clients cached by the previous test."""
    harness.net.reset()
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield harness.net
    litellm.in_memory_llm_clients_cache.flush_cache()
